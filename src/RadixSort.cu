#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <iostream>

#include "RadixSort.cuh"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

__device__ void d_prefix_sum(unsigned int* values, unsigned int n) {
  int offset = 1;

  for (int d = n >> 1; d > 0; d >>= 1) {
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
      int ai = offset * (2 * i + 1) - 1;
      int bi = offset * (2 * i + 2) - 1;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      values[bi] += values[ai];
    }
    offset <<= 1;
  }

  if (!threadIdx.x) {
    int last_idx = n - 1;
    values[last_idx + CONFLICT_FREE_OFFSET(last_idx)] = 0;
  }

  for (int d = 1; d < n; d *= 2) {
    offset >>= 1;
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
      int ai = offset * (2 * i + 1) - 1;
      int bi = offset * (2 * i + 2) - 1;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      unsigned int t = values[ai];
      values[ai] = values[bi];
      values[bi] += t;
    }
  }
}

__global__ void radixSetup(unsigned int* radices, unsigned int* cells,
                           const unsigned int shift,
                           ConstantsRadixSetUp constants) {
  extern __shared__ unsigned int s[];
  int l_group = threadIdx.x / constants.threads_per_group;
  int group_start = (blockIdx.x * constants.groups_per_block + l_group) *
                    constants.cells_per_group;
  int group_end = group_start + constants.cells_per_group;

  int padded_radices = constants.num_radices + 2;

  for (int i = threadIdx.x; i < constants.shared_memory_size; i += blockDim.x) {
    s[i] = 0;
  }
  __syncthreads();

  for (int i = group_start + (threadIdx.x % constants.threads_per_group);
       (i < group_end) && (i < constants.cells_size);
       i += constants.threads_per_group) {
    unsigned int digit = (cells[i] >> shift) & (constants.num_radices - 1);
    int index = l_group * padded_radices + digit;
    atomicAdd(&s[index], 1);
  }
  __syncthreads();

  for (int i = threadIdx.x;
       i < constants.groups_per_block * constants.num_radices;
       i += blockDim.x) {
    int bucket = i / constants.groups_per_block;
    int local_group = i % constants.groups_per_block;
    int global_group = (blockIdx.x * constants.groups_per_block) + local_group;

    unsigned int val = s[local_group * padded_radices + bucket];
    radices[(bucket * gridDim.x * constants.groups_per_block) + global_group] =
        val;
  }
}

__global__ void radixSum(unsigned int* radices,
                         unsigned int* radices_prefix_sum,
                         ConstantsRadixSum constants) {
  extern __shared__ unsigned int s[];
  int left = 0;
  int total;

  for (int j = 0;
       j < constants.radices_per_block &&
       (blockIdx.x * constants.radices_per_block + j) < constants.num_radices;
       j++) {
    for (int i = threadIdx.x; i < constants.num_groups; i += blockDim.x) {
      int padded_i = i + CONFLICT_FREE_OFFSET(i);
      s[padded_i] = radices[blockIdx.x * constants.num_groups *
                                constants.radices_per_block +
                            j * constants.num_groups + i];
    }
    __syncthreads();

    for (int i = threadIdx.x + constants.num_groups;
         i < constants.padded_groups; i += blockDim.x) {
      int padded_i = i + CONFLICT_FREE_OFFSET(i);
      s[padded_i] = 0;
    }
    __syncthreads();

    if (!threadIdx.x) {
      total = s[constants.num_groups - 1 +
                CONFLICT_FREE_OFFSET(constants.num_groups - 1)];
    }

    d_prefix_sum(s, constants.padded_groups);
    __syncthreads();

    for (int i = threadIdx.x; i < constants.num_groups; i += blockDim.x) {
      int padded_i = i + CONFLICT_FREE_OFFSET(i);
      radices[blockIdx.x * constants.num_groups * constants.radices_per_block +
              j * constants.num_groups + i] = s[padded_i];
    }
    __syncthreads();

    if (!threadIdx.x) {
      total += s[constants.num_groups - 1 +
                 CONFLICT_FREE_OFFSET(constants.num_groups - 1)];
      radices_prefix_sum[blockIdx.x * constants.radices_per_block + j] = left;
      left += total;
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    radices_prefix_sum[constants.num_radices + blockIdx.x] = left;
  }
}

__global__ void radixReorder(unsigned int* radices_prefix_sum,
                             unsigned int* radices, unsigned int* cells,
                             unsigned int* objects, unsigned int* cells_out,
                             unsigned int* objects_out,
                             const unsigned int shift,
                             ConstantsRadixReorder constants) {
  extern __shared__ unsigned int s[];
  unsigned int* t = s + constants.num_radices;
  int l_group = threadIdx.x / constants.threads_per_group;
  int group_start = (blockIdx.x * constants.groups_per_block + l_group) *
                    constants.cells_per_group;
  int group_end = group_start + constants.cells_per_group;

  for (int i = threadIdx.x; i < constants.num_radices; i += blockDim.x) {
    s[i] = radices_prefix_sum[i];
    if (i < constants.sum_blocks) {
      int padded_i = i + CONFLICT_FREE_OFFSET(i);
      t[padded_i] = radices_prefix_sum[constants.num_radices + i];
    }
  }
  __syncthreads();

  for (int i = threadIdx.x + constants.sum_blocks; i < constants.padded_blocks;
       i += blockDim.x) {
    int padded_i = i + CONFLICT_FREE_OFFSET(i);
    t[padded_i] = 0;
  }
  __syncthreads();

  d_prefix_sum(t, constants.padded_blocks);
  __syncthreads();

  for (int i = threadIdx.x; i < constants.num_radices; i += blockDim.x) {
    int padded_idx = (i / constants.radices_per_block) +
                     CONFLICT_FREE_OFFSET(i / constants.radices_per_block);
    s[i] += t[padded_idx];
  }
  __syncthreads();

  int padded_radices = constants.num_radices + 2;

  for (int i = threadIdx.x;
       i < constants.groups_per_block * constants.num_radices;
       i += blockDim.x) {
    int bucket = i % constants.num_radices;
    int local_group = i / constants.num_radices;
    int global_group = (blockIdx.x * constants.groups_per_block) + local_group;

    t[local_group * padded_radices + bucket] =
        s[bucket] + radices[((bucket)*gridDim.x * constants.groups_per_block) +
                            global_group];
  }
  __syncthreads();

  for (int i = group_start + (threadIdx.x % constants.threads_per_group);
       (i < group_end) && (i < constants.cells_size);
       i += constants.threads_per_group) {
    unsigned int digit = (cells[i] >> shift) & (constants.num_radices - 1);
    int index = l_group * padded_radices + digit;

    unsigned int target_idx = atomicAdd(&t[index], 1);

    cells_out[target_idx] = cells[i];
    objects_out[target_idx] = objects[i];
  }
}

unsigned int RadixSort::succesive_power_of_two(unsigned int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

RadixSort::RadixSort()
    : d_cells(nullptr),
      d_objects(nullptr),
      d_cells_tmp(nullptr),
      d_objects_tmp(nullptr),
      d_radices(nullptr),
      d_radices_prefix_sum(nullptr),
      array_size(0),
      min_bits_for_hash(0),
      num_radices(256),
      threads_per_group(32) {}

RadixSort::~RadixSort() {
  if (d_cells_tmp) cudaFree(d_cells_tmp);
  if (d_objects_tmp) cudaFree(d_objects_tmp);
  if (d_radices) cudaFree(d_radices);
  if (d_radices_prefix_sum) cudaFree(d_radices_prefix_sum);
}

void RadixSort::setup(unsigned int array_size, unsigned int min_bits_for_hash,
                      unsigned int* d_cells, unsigned int* d_objects) {
  this->min_bits_for_hash = min_bits_for_hash;
  this->array_size = array_size;
  this->d_cells = d_cells;
  this->d_objects = d_objects;
  this->num_radices = 256;
  this->threads_per_group = 32;

  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, deviceId);

  int minGridMain, bSizeMain;

  int max_prefix_padding = this->num_radices >> LOG_NUM_BANKS;

  auto smemReorder = [=](int bSize) -> size_t {
    int groups = bSize / this->threads_per_group;
    return (this->num_radices + (groups * (this->num_radices + 2)) +
            max_prefix_padding) *
           sizeof(unsigned int);
  };

  int optimal_bSizeMain = 128;
  int max_occupancy = 0;

  for (int bs = 128; bs <= 1024; bs += 32) {
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, radixReorder, bs,
                                                  smemReorder(bs));
    int occupancy = numBlocks * bs;
    if (occupancy > max_occupancy) {
      max_occupancy = occupancy;
      optimal_bSizeMain = bs;
    }
  }

  cudaOccupancyMaxPotentialBlockSizeVariableSMem(
      &minGridMain, &bSizeMain, radixReorder, smemReorder, optimal_bSizeMain);

  this->num_threads_main = optimal_bSizeMain;
  this->num_blocks_main = minGridMain;

  if (this->num_blocks_main < 2 * device_prop.multiProcessorCount) {
    this->num_blocks_main = 2 * device_prop.multiProcessorCount;
  }

  int groups_per_block_main = this->num_threads_main / this->threads_per_group;
  int num_groups = this->num_blocks_main * groups_per_block_main;
  int cells_per_group = (this->array_size - 1) / num_groups + 1;

  this->setup_smem_bytes =
      groups_per_block_main * (this->num_radices + 2) * sizeof(unsigned int);

  int minGridSum, bSizeSum;
  int padded_groups = succesive_power_of_two(num_groups);

  auto smemSum = [=](int bSize) -> size_t {
    int padding = padded_groups >> LOG_NUM_BANKS;
    return (padded_groups + padding) * sizeof(unsigned int);
  };

  cudaOccupancyMaxPotentialBlockSizeVariableSMem(
      &minGridSum, &bSizeSum, radixSum, smemSum, optimal_bSizeMain);

  this->num_threads_sum = bSizeSum;

  this->num_blocks_sum = std::min((int)this->num_radices, minGridSum);

  int padded_blocks_sum = succesive_power_of_two(this->num_blocks_sum);
  int radices_per_block = (this->num_radices - 1) / this->num_blocks_sum + 1;

  int sum_padding = padded_groups >> LOG_NUM_BANKS;
  this->sum_smem_bytes = (padded_groups + sum_padding) * sizeof(unsigned int);

  int reorder_prefix_padding = padded_blocks_sum >> LOG_NUM_BANKS;
  this->reorder_smem_bytes =
      (this->num_radices + (groups_per_block_main * (this->num_radices + 2)) +
       reorder_prefix_padding) *
      sizeof(unsigned int);

  constantsRadixSetUp.groups_per_block = groups_per_block_main;
  constantsRadixSetUp.threads_per_group = this->threads_per_group;
  constantsRadixSetUp.cells_per_group = cells_per_group;
  constantsRadixSetUp.shared_memory_size =
      groups_per_block_main * (this->num_radices + 2);
  constantsRadixSetUp.num_radices = this->num_radices;
  constantsRadixSetUp.cells_size = this->array_size;

  constantsRadixSum.num_groups = num_groups;
  constantsRadixSum.num_radices = this->num_radices;
  constantsRadixSum.padded_groups = padded_groups;
  constantsRadixSum.radices_per_block = radices_per_block;

  constantsRadixReorder.cells_per_group = cells_per_group;
  constantsRadixReorder.cells_size = this->array_size;
  constantsRadixReorder.groups_per_block = groups_per_block_main;
  constantsRadixReorder.num_radices = this->num_radices;
  constantsRadixReorder.padded_blocks = padded_blocks_sum;
  constantsRadixReorder.radices_per_block = radices_per_block;
  constantsRadixReorder.threads_per_group = this->threads_per_group;
  constantsRadixReorder.sum_blocks = this->num_blocks_sum;

  cudaMalloc(&this->d_radices,
             sizeof(unsigned int) * num_groups * this->num_radices);
  cudaMalloc(&this->d_radices_prefix_sum,
             sizeof(unsigned int) * (this->num_radices + this->num_blocks_sum));
  cudaMalloc(&this->d_cells_tmp, sizeof(unsigned int) * this->array_size);
  cudaMalloc(&this->d_objects_tmp, sizeof(unsigned int) * this->array_size);
}

void RadixSort::sort() {
  unsigned int* current_cells = this->d_cells;
  unsigned int* current_objects = this->d_objects;
  unsigned int* tmp_cells = this->d_cells_tmp;
  unsigned int* tmp_objects = this->d_objects_tmp;

  for (int shift = 0; shift < this->min_bits_for_hash; shift += 8) {
    radixSetup<<<this->num_blocks_main, this->num_threads_main,
                 this->setup_smem_bytes>>>(this->d_radices, current_cells,
                                           shift, this->constantsRadixSetUp);

    radixSum<<<this->num_blocks_sum, this->num_threads_sum,
               this->sum_smem_bytes>>>(
        this->d_radices, this->d_radices_prefix_sum, this->constantsRadixSum);

    radixReorder<<<this->num_blocks_main, this->num_threads_main,
                   this->reorder_smem_bytes>>>(
        this->d_radices_prefix_sum, this->d_radices, current_cells,
        current_objects, tmp_cells, tmp_objects, shift,
        this->constantsRadixReorder);

    std::swap(current_cells, tmp_cells);
    std::swap(current_objects, tmp_objects);
  }

  if (current_cells != this->d_cells) {
    cudaMemcpy(this->d_cells, current_cells,
               sizeof(unsigned int) * this->array_size,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(this->d_objects, current_objects,
               sizeof(unsigned int) * this->array_size,
               cudaMemcpyDeviceToDevice);
  }
}