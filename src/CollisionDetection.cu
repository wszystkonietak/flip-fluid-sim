#include <math.h>

#include <algorithm>
#include <iostream>

#include "CollisionDetection.cuh"

struct ConstantsInitData {
  float diameter;
  float radius;
  unsigned int x_shift;
  unsigned int objects_size;
  unsigned int width;
  unsigned int height;
};

struct ConstantsCountCollisionCells {
  unsigned int cell_count;
  unsigned int cells_per_thread;
  unsigned int x_shift;
  unsigned int max_grid_capacity;
};


__device__ void dSum(unsigned int* values, unsigned int* out) {
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      values[threadIdx.x] += values[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(out, values[0]);
  }
}

__device__ void processPair(unsigned int objA_id, unsigned int objB_id,
                            float2& posA, float2& posB, float2* positions,
                            float radius, float dist_threshold,
                            unsigned int& local_count) {
  float dx = posA.x - posB.x;
  float dy = posA.y - posB.y;

  float dist_sq = __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));

  if (dist_sq <= dist_threshold) {
    local_count++;

    if (dist_sq > 1e-8f) {
      float dist = sqrtf(dist_sq);
      dx /= dist;
      dy /= dist;

      float push_mag = (2.0001f * radius - dist) / 2.0f;

      posA.x += dx * push_mag;
      posA.y += dy * push_mag;

      posB.x -= dx * push_mag;
      posB.y -= dy * push_mag;

      positions[objA_id] = posA;
      positions[objB_id] = posB;
    } else {
      posA.x += radius;
      posB.x -= radius;
      posB.y += radius;
      posA.y -= radius;

      positions[objA_id] = posA;
      positions[objB_id] = posB;
    }
  }
}

__global__ void applyBoundaries(float2* positions, float scene_width,
                                float scene_height, float radius,
                                unsigned int objects_size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < objects_size) {
    float2 pos = positions[idx];

    pos.x = max(radius, min(scene_width - radius, pos.x));
    pos.y = max(radius, min(scene_height - radius, pos.y));

    positions[idx] = pos;
  }
}

__global__ void initData(float2* positions, unsigned int* cells,
                         unsigned int* objects, unsigned int* cell_count,
                         unsigned int* control_bits,
                         ConstantsInitData constants) {
  extern __shared__ unsigned int s[];
  unsigned int count = 0;
  unsigned int g_id = blockIdx.x * blockDim.x + threadIdx.x;

  while (g_id < constants.objects_size) {
    count++;
    float pos_x = positions[g_id].x;
    float pos_y = positions[g_id].y;

    int cell_x = max(0, min((int)constants.width - 1,
                            (int)floorf(pos_x / constants.diameter)));
    int cell_y = max(0, min((int)constants.height - 1,
                            (int)floorf(pos_y / constants.diameter)));
    unsigned int type = (cell_x & 1) | ((cell_y & 1) << 1);

    unsigned int mask = (1 << type);

    bool right = false, left = false, up = false, down = false;
    if (pos_x + constants.radius > (cell_x + 1) * constants.diameter &&
        (cell_x + 1) < constants.width)
      right = true;
    else if (pos_x - constants.radius < cell_x * constants.diameter &&
             cell_x > 0)
      left = true;

    if (pos_y + constants.radius > (cell_y + 1) * constants.diameter &&
        (cell_y + 1) < constants.height)
      up = true;
    else if (pos_y - constants.radius < cell_y * constants.diameter &&
             cell_y > 0)
      down = true;

    int boundaries =
        (right ? 1 : 0) + (left ? 1 : 0) + (up ? 1 : 0) + (down ? 1 : 0);
    int addValX = right ? 1 : (left ? -1 : 0);
    int addValY = up ? 1 : (down ? -1 : 0);

    if (right)
      mask |= (1 << (((cell_x + 1) & 1) | ((cell_y & 1) << 1)));
    else if (left)
      mask |= (1 << (((cell_x - 1) & 1) | ((cell_y & 1) << 1)));
    if (up)
      mask |= (1 << ((cell_x & 1) | (((cell_y + 1) & 1) << 1)));
    else if (down)
      mask |= (1 << ((cell_x & 1) | (((cell_y - 1) & 1) << 1)));
    if (boundaries == 2)
      mask |=
          (1 << (((cell_x + addValX) & 1) | (((cell_y + addValY) & 1) << 1)));

    control_bits[g_id] = (mask << 2) | type;

    cells[4 * g_id] =
        ((((unsigned int)cell_x << constants.x_shift) | (unsigned int)cell_y)
         << 1) |
        0x00;
    objects[4 * g_id] = g_id << 1 | 0x01;

    unsigned int l_id = 1;

    if (boundaries == 2) {
      if (right || left) {
        cells[4 * g_id + l_id] =
            ((((unsigned int)(cell_x + addValX) << constants.x_shift) |
              (unsigned int)cell_y)
             << 1) |
            0x01;
        objects[4 * g_id + l_id] = g_id << 1 | 0x00;
        l_id++;
        count++;
      }
      if (up || down) {
        cells[4 * g_id + l_id] = ((((unsigned int)cell_x << constants.x_shift) |
                                   (unsigned int)(cell_y + addValY))
                                  << 1) |
                                 0x01;
        objects[4 * g_id + l_id] = g_id << 1 | 0x00;
        l_id++;
        count++;
      }
      unsigned int diag_x = cell_x + addValX;
      unsigned int diag_y = cell_y + addValY;
      cells[4 * g_id + l_id] =
          (((diag_x << constants.x_shift) | diag_y) << 1) | 0x01;
      objects[4 * g_id + l_id] = g_id << 1 | 0x00;
      l_id++;
      count++;

    } else if (boundaries == 1) {
      if (right || left) {
        cells[4 * g_id + l_id] =
            ((((unsigned int)(cell_x + addValX) << constants.x_shift) |
              (unsigned int)cell_y)
             << 1) |
            0x01;
        objects[4 * g_id + l_id] = g_id << 1 | 0x00;
        l_id++;
        count++;

        if (cell_y + 1 < constants.height) {
          cells[4 * g_id + l_id] =
              ((((unsigned int)(cell_x + addValX) << constants.x_shift) |
                (unsigned int)(cell_y + 1))
               << 1) |
              0x01;
          objects[4 * g_id + l_id] = g_id << 1 | 0x00;
          l_id++;
          count++;
        }
        if (cell_y > 0) {
          cells[4 * g_id + l_id] =
              ((((unsigned int)(cell_x + addValX) << constants.x_shift) |
                (unsigned int)(cell_y - 1))
               << 1) |
              0x01;
          objects[4 * g_id + l_id] = g_id << 1 | 0x00;
          l_id++;
          count++;
        }
      } else if (up || down) {
        cells[4 * g_id + l_id] = ((((unsigned int)cell_x << constants.x_shift) |
                                   (unsigned int)(cell_y + addValY))
                                  << 1) |
                                 0x01;
        objects[4 * g_id + l_id] = g_id << 1 | 0x00;
        l_id++;
        count++;

        if (cell_x + 1 < constants.width) {
          cells[4 * g_id + l_id] =
              ((((unsigned int)(cell_x + 1) << constants.x_shift) |
                (unsigned int)(cell_y + addValY))
               << 1) |
              0x01;
          objects[4 * g_id + l_id] = g_id << 1 | 0x00;
          l_id++;
          count++;
        }
        if (cell_x > 0) {
          cells[4 * g_id + l_id] =
              ((((unsigned int)(cell_x - 1) << constants.x_shift) |
                (unsigned int)(cell_y + addValY))
               << 1) |
              0x01;
          objects[4 * g_id + l_id] = g_id << 1 | 0x00;
          l_id++;
          count++;
        }
      }
    }

    for (unsigned int pad = l_id; pad < 4; pad++) {
      cells[4 * g_id + pad] = UINT_MAX;
      objects[4 * g_id + pad] = g_id << 2;
    }

    g_id += blockDim.x * gridDim.x;
  }

  s[threadIdx.x] = count;
  __syncthreads();
  dSum(s, cell_count);
}

__global__ void countCollisionCells(unsigned int* cells, unsigned int* objects,
                                    uint3* collision_cells_uncompacted,
                                    unsigned int* collision_cells_ids,
                                    ConstantsCountCollisionCells constants) {
  unsigned int thread_start =
      (blockIdx.x * blockDim.x + threadIdx.x) * constants.cells_per_thread;
  unsigned int thread_end = thread_start + constants.cells_per_thread;
  unsigned int last = UINT_MAX;
  unsigned int i = thread_start;

  unsigned int home_cell_count;
  unsigned int phantom_cell_count;
  int cell_start_index = -1;

  while (1) {
    if (i >= constants.cell_count || cells[i] >> 1 != last) {
      if (cell_start_index + 1) {
        collision_cells_uncompacted[last].x = cell_start_index;
        collision_cells_uncompacted[last].y = home_cell_count;
        collision_cells_uncompacted[last].z = phantom_cell_count;

        if (home_cell_count > 0 && (home_cell_count + phantom_cell_count) > 1) {
          unsigned int cell_y = last & ((1 << constants.x_shift) - 1);
          unsigned int cell_x = last >> constants.x_shift;
          unsigned int list_id = (cell_x & 1) | ((cell_y & 1) << 1);
          collision_cells_ids[list_id * constants.max_grid_capacity + last] = 1;
        }
      }

      if (i > thread_end || i >= constants.cell_count) break;

      if (i != thread_start || (!blockIdx.x && !threadIdx.x)) {
        home_cell_count = 0;
        phantom_cell_count = 0;
        cell_start_index = i;
      }
      last = cells[i] >> 1;
    }

    if (cell_start_index + 1) {
      if (objects[i] & 0x01)
        home_cell_count++;
      else
        phantom_cell_count++;
    }
    i++;
  }
}

__global__ void compactCollisionCells(uint3* d_uncompacted,
                                      unsigned int* d_scanned_ids,
                                      uint3* d_compacted_cells,
                                      unsigned int max_grid_capacity,
                                      unsigned int x_shift) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_grid_capacity) return;

  uint3 cell_data = d_uncompacted[idx];
  unsigned int home = cell_data.y;
  unsigned int phantom = cell_data.z;

  if (home > 0 && (home + phantom) > 1) {
    unsigned int cell_y = idx & ((1 << x_shift) - 1);
    unsigned int cell_x = idx >> x_shift;
    unsigned int list_id = (cell_x & 1) | ((cell_y & 1) << 1);

    unsigned int dest_idx = d_scanned_ids[list_id * max_grid_capacity + idx];
    d_compacted_cells[dest_idx] = cell_data;
  }
}

__global__ void resolveCollisions(uint3* compacted_cells,
                                  unsigned int* sorted_objects,
                                  float2* positions, float diameter,
                                  float radius, unsigned int x_shift,
                                  unsigned int start_idx, unsigned int end_idx,
                                  unsigned int pass_T,
                                  unsigned int* d_collisions_count) {
  extern __shared__ unsigned int s_collisions[];
  unsigned int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int local_count = 0;

  if (idx < end_idx) {
    uint3 cell_data = compacted_cells[idx];
    unsigned int cell_start = cell_data.x;
    unsigned int home_count = cell_data.y;
    unsigned int phantom_count = cell_data.z;
    unsigned int total_count = home_count + phantom_count;

    float dist_threshold = 4.0f * radius * radius;

    for (unsigned int i = 0; i < home_count; ++i) {
      unsigned int raw_objA = sorted_objects[cell_start + i];
      unsigned int objA_id = raw_objA >> 1;
      float2 posA = positions[objA_id];

      for (unsigned int j = i + 1; j < total_count; ++j) {
        unsigned int raw_objB = sorted_objects[cell_start + j];
        unsigned int objB_id = raw_objB >> 1;
        float2 posB = positions[objB_id];

        processPair(objA_id, objB_id, posA, posB, positions, radius,
                    dist_threshold, local_count);
      }
    }
  }

  s_collisions[threadIdx.x] = local_count;
  __syncthreads();
  dSum(s_collisions, d_collisions_count);
}

unsigned int CollisionDetection::count_bits(unsigned int n) {
  unsigned int count = 0;
  while (n) {
    count++;
    n >>= 1;
  }
  return count;
}

CollisionDetection::~CollisionDetection() {
  if (d_cells) cudaFree(d_cells);
  if (d_objects) cudaFree(d_objects);
  if (d_cell_count) cudaFree(d_cell_count);
  if (d_collision_cells_uncompacted) cudaFree(d_collision_cells_uncompacted);
  if (d_collision_cells_ids) cudaFree(d_collision_cells_ids);
  if (d_collisions_count) cudaFree(d_collisions_count);
  if (d_collision_cells) cudaFree(d_collision_cells);
  if (d_control_bits) cudaFree(d_control_bits);
}

void CollisionDetection::setup(float scene_width, float scene_height,
                               unsigned int size, float radius) {
  this->scene_width = scene_width;
  this->scene_height = scene_height;
  this->objects_size = size;
  this->radius = radius;

  this->diameter = 3.0f * radius;

  this->width = ceilf(scene_width / diameter) + 1;
  this->height = ceilf(scene_height / diameter) + 1;

  unsigned int min_bits_for_width = count_bits(width);
  unsigned int min_bits_for_height = count_bits(height);

  this->min_bits_for_hash = 32;
  this->x_shift = min_bits_for_height;
  this->max_grid_capacity = width * (1 << x_shift);
  this->cells_size = 4 * objects_size;

  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, deviceId);

  this->num_blocks = 2 * device_prop.multiProcessorCount;
  this->num_threads = 256;
  this->shared_size = num_threads * sizeof(unsigned int);

  cudaMalloc((void**)&d_cells, sizeof(unsigned int) * cells_size);
  cudaMalloc((void**)&d_objects, sizeof(unsigned int) * cells_size);
  cudaMalloc((void**)&d_cell_count, sizeof(unsigned int));
  cudaMalloc((void**)&d_collisions_count, sizeof(unsigned int));
  cudaMalloc((void**)&d_collision_cells_uncompacted,
             sizeof(uint3) * max_grid_capacity);
  cudaMalloc((void**)&d_collision_cells_ids,
             sizeof(unsigned int) * (4 * max_grid_capacity + 1));
  cudaMalloc((void**)&d_collision_cells, sizeof(uint3) * max_grid_capacity);
  cudaMalloc((void**)&d_control_bits, sizeof(unsigned int) * objects_size);

  sorter.setup(cells_size, min_bits_for_hash, d_cells, d_objects);
}

void CollisionDetection::check_collision(float2* d_positions) {
  cudaMemset(d_cell_count, 0, sizeof(unsigned int));
  cudaMemset(d_collisions_count, 0, sizeof(unsigned int));
  cudaMemset(d_collision_cells_uncompacted, 0,
             sizeof(uint3) * max_grid_capacity);

  cudaMemset(d_collision_cells_ids, 0,
             sizeof(unsigned int) * (4 * max_grid_capacity + 1));

  cudaMemset(d_collision_cells, 0, sizeof(uint3) * max_grid_capacity);

  ConstantsInitData constantsInitData;
  constantsInitData.diameter = diameter;
  constantsInitData.radius = radius;
  constantsInitData.x_shift = x_shift;
  constantsInitData.objects_size = objects_size;
  constantsInitData.width = width;
  constantsInitData.height = height;

  initData<<<num_blocks, num_threads, shared_size>>>(
      d_positions, d_cells, d_objects, d_cell_count, d_control_bits,
      constantsInitData);

  sorter.sort();

  cudaMemcpy(&h_cell_count, d_cell_count, sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  unsigned int cells_per_thread =
      (h_cell_count - 1) / (num_blocks * num_threads) + 1;

  ConstantsCountCollisionCells constantsCountCollisionCells;
  constantsCountCollisionCells.cell_count = h_cell_count;
  constantsCountCollisionCells.cells_per_thread = cells_per_thread;
  constantsCountCollisionCells.x_shift = x_shift;
  constantsCountCollisionCells.max_grid_capacity = max_grid_capacity;

  countCollisionCells<<<num_blocks, num_threads>>>(
      d_cells, d_objects, d_collision_cells_uncompacted, d_collision_cells_ids,
      constantsCountCollisionCells);

  runGlobalPrefixSum(d_collision_cells_ids, 4 * max_grid_capacity + 1);

  unsigned int compact_threads = 256;
  unsigned int compact_blocks =
      (max_grid_capacity + compact_threads - 1) / compact_threads;

  compactCollisionCells<<<compact_blocks, compact_threads>>>(
      d_collision_cells_uncompacted, d_collision_cells_ids, d_collision_cells,
      max_grid_capacity, x_shift);

  unsigned int pass_endpoints[5];
  pass_endpoints[0] = 0;
  cudaMemcpy(&pass_endpoints[1], &d_collision_cells_ids[1 * max_grid_capacity],
             sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&pass_endpoints[2], &d_collision_cells_ids[2 * max_grid_capacity],
             sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&pass_endpoints[3], &d_collision_cells_ids[3 * max_grid_capacity],
             sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&pass_endpoints[4], &d_collision_cells_ids[4 * max_grid_capacity],
             sizeof(unsigned int), cudaMemcpyDeviceToHost);

  for (unsigned int pass_T = 0; pass_T < 4; ++pass_T) {
    unsigned int start_idx = pass_endpoints[pass_T];
    unsigned int end_idx = pass_endpoints[pass_T + 1];
    unsigned int cells_in_pass = end_idx - start_idx;

    if (cells_in_pass > 0) {
      unsigned int resolve_threads = 256;
      unsigned int resolve_blocks =
          (cells_in_pass + resolve_threads - 1) / resolve_threads;
      unsigned int resolve_shared_size = resolve_threads * sizeof(unsigned int);

      resolveCollisions<<<resolve_blocks, resolve_threads,
                          resolve_shared_size>>>(
          d_collision_cells, d_objects, d_positions, diameter, radius, x_shift,
          start_idx, end_idx, pass_T, d_collisions_count);
    }
  }
  unsigned int boundary_threads = 256;
  unsigned int boundary_blocks =
      (objects_size + boundary_threads - 1) / boundary_threads;
  applyBoundaries<<<boundary_blocks, boundary_threads>>>(
      d_positions, scene_width, scene_height, radius, objects_size);

  cudaMemcpy(&h_gpu_collisions, d_collisions_count, sizeof(unsigned int),
             cudaMemcpyDeviceToHost);
}

unsigned int CollisionDetection::get_gpu_collisions_count() const {
  return h_gpu_collisions;
}

void CollisionDetection::get_sorted_data(std::vector<unsigned int>& h_cells,
                                         std::vector<unsigned int>& h_objects,
                                         unsigned int& out_cell_count) const {
  out_cell_count = this->h_cell_count;
  if (h_cells.size() < cells_size) h_cells.resize(cells_size);
  if (h_objects.size() < cells_size) h_objects.resize(cells_size);

  cudaMemcpy(h_cells.data(), d_cells, sizeof(unsigned int) * cells_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_objects.data(), d_objects, sizeof(unsigned int) * cells_size,
             cudaMemcpyDeviceToHost);
}
