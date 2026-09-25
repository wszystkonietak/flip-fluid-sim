#include "Water.cuh"

__global__ void create_solid_cells(float* solid_cells, int width, int height) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    float is_fluid =
        (x < 1 || x > width - 2 || y < 1 || y > height - 2) ? 0.0f : 1.0f;
    solid_cells[x * height + y] = is_fluid;
  }
}

__global__ void simulate_particles(
    float* grid, Particles particles, float2* grid_velocities,
    float2* sum_of_weights, ushort2* busy_cells, unsigned int* busy_cells_size,
    unsigned int particles_size, uint2 resolution, unsigned int grid_size,
    float4 boundings, float cell_size, float delta_time,
    float4 interaction_data, bool is_interacting) {
  int g_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (g_id >= particles_size) return;

  float2 vel = particles.velocities[g_id];
  float2 pos = particles.positions[g_id];

  if (is_interacting) {
    float dx = pos.x - interaction_data.x;
    float dy = pos.y - interaction_data.y;
    float dist_sq = (dx * dx) + (dy * dy);

    float interaction_radius = 0.1f;
    float force_multiplier = 100.0f;

    float radius_sq = interaction_radius * interaction_radius;

    if (dist_sq < radius_sq) {
      float weight = 1.0f - (dist_sq / radius_sq);

      vel.x += interaction_data.z * weight * force_multiplier * delta_time;
      vel.y += interaction_data.w * weight * force_multiplier * delta_time;
    }
  }

  vel.y -= 5.81f * delta_time;
  pos.x += vel.x * delta_time;
  pos.y += vel.y * delta_time;

  if (pos.x <= boundings.x) {
    vel.x *= -0.8f;
    pos.x = boundings.x;
  }
  if (pos.x >= boundings.y) {
    vel.x *= -0.8f;
    pos.x = boundings.y;
  }
  if (pos.y <= boundings.z) {
    vel.y *= -0.8f;
    pos.y = boundings.z;
  }
  if (pos.y >= boundings.w) {
    vel.y *= -0.8f;
    pos.y = boundings.w;
  }

  particles.positions[g_id] = pos;
  particles.velocities[g_id] = vel;

  int c_x = max(0, min((int)(pos.x / cell_size), (int)resolution.x - 1));
  int c_y = max(0, min((int)(pos.y / cell_size), (int)resolution.y - 1));
  grid[c_x * resolution.y + c_y] = 1.0f;

  float px_u = pos.x / cell_size;
  float py_u = (pos.y / cell_size) - 0.5f;
  int cx_u = max(0, min((int)px_u, (int)resolution.x - 2));
  int cy_u = max(0, min((int)py_u, (int)resolution.y - 2));
  float tx_u = px_u - cx_u;
  float ty_u = py_u - cy_u;

  float4 w_u = make_float4((1 - tx_u) * (1 - ty_u), tx_u * (1 - ty_u),
                           tx_u * ty_u, (1 - tx_u) * ty_u);

  atomicAdd(&grid_velocities[cx_u * resolution.y + cy_u].x, w_u.x * vel.x);
  atomicAdd(&grid_velocities[(cx_u + 1) * resolution.y + cy_u].x,
            w_u.y * vel.x);
  atomicAdd(&grid_velocities[(cx_u + 1) * resolution.y + cy_u + 1].x,
            w_u.z * vel.x);
  atomicAdd(&grid_velocities[cx_u * resolution.y + cy_u + 1].x, w_u.w * vel.x);

  atomicAdd(&sum_of_weights[cx_u * resolution.y + cy_u].x, w_u.x);
  atomicAdd(&sum_of_weights[(cx_u + 1) * resolution.y + cy_u].x, w_u.y);
  atomicAdd(&sum_of_weights[(cx_u + 1) * resolution.y + cy_u + 1].x, w_u.z);
  atomicAdd(&sum_of_weights[cx_u * resolution.y + cy_u + 1].x, w_u.w);

  float px_v = (pos.x / cell_size) - 0.5f;
  float py_v = pos.y / cell_size;
  int cx_v = max(0, min((int)px_v, (int)resolution.x - 2));
  int cy_v = max(0, min((int)py_v, (int)resolution.y - 2));
  float tx_v = px_v - cx_v;
  float ty_v = py_v - cy_v;

  float4 w_v = make_float4((1 - tx_v) * (1 - ty_v), tx_v * (1 - ty_v),
                           tx_v * ty_v, (1 - tx_v) * ty_v);

  atomicAdd(&grid_velocities[cx_v * resolution.y + cy_v].y, w_v.x * vel.y);
  atomicAdd(&grid_velocities[(cx_v + 1) * resolution.y + cy_v].y,
            w_v.y * vel.y);
  atomicAdd(&grid_velocities[(cx_v + 1) * resolution.y + cy_v + 1].y,
            w_v.z * vel.y);
  atomicAdd(&grid_velocities[cx_v * resolution.y + cy_v + 1].y, w_v.w * vel.y);

  atomicAdd(&sum_of_weights[cx_v * resolution.y + cy_v].y, w_v.x);
  atomicAdd(&sum_of_weights[(cx_v + 1) * resolution.y + cy_v].y, w_v.y);
  atomicAdd(&sum_of_weights[(cx_v + 1) * resolution.y + cy_v + 1].y, w_v.z);
  atomicAdd(&sum_of_weights[cx_v * resolution.y + cy_v + 1].y, w_v.w);
}

__global__ void update_velocities(float2* grid_velocities,
                                  float2* sum_of_weights, ushort2* busy_cells,
                                  unsigned int* busy_cells_size,
                                  uint2 resolution) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int idx = x * resolution.y + y;
    float2 sum = sum_of_weights[idx];
    float2 vel = grid_velocities[idx];

    if (sum.x > 0.0f) vel.x /= sum.x;
    if (sum.y > 0.0f) vel.y /= sum.y;

    grid_velocities[idx] = vel;
  }
}

__global__ void calculate_divergence(float* grid, float* solid_cells,
                                     float2* grid_velocities, uint2 resolution,
                                     unsigned int iteration) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (x < resolution.x - 1 && y < resolution.y - 1) {
    float isBusy = grid[x * resolution.y + y];
    if (isBusy == 1.0f) {
      if ((x + y) % 2 == (iteration % 2)) {
        float left = grid_velocities[((x)*resolution.y) + y].x;
        float right = grid_velocities[((x + 1) * resolution.y) + y].x;
        float bottom = grid_velocities[(x * resolution.y) + (y)].y;
        float top = grid_velocities[(x * resolution.y) + (y + 1)].y;

        float solid_left = solid_cells[(x - 1) * resolution.y + y];
        float solid_right = solid_cells[(x + 1) * resolution.y + y];
        float solid_bottom = solid_cells[x * resolution.y + (y - 1)];
        float solid_top = solid_cells[x * resolution.y + (y + 1)];

        float sumOfStates = solid_left + solid_right + solid_bottom + solid_top;
        float divergence = (right - left + top - bottom);

        divergence *= 1.9f;
        divergence /= sumOfStates;
        grid_velocities[((x + 1) * resolution.y) + y].x -=
            divergence * solid_right;
        grid_velocities[((x)*resolution.y) + y].x += divergence * solid_left;
        grid_velocities[(x * resolution.y) + (y + 1)].y -=
            divergence * solid_top;
        grid_velocities[(x * resolution.y) + (y)].y +=
            divergence * solid_bottom;
      }
    }
  }
}

__global__ void grid_to_particles(float2* grid_velocities, Particles particles,
                                  float cell_size, int num_particles,
                                  uint2 resolution) {
  int g_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (g_id >= num_particles) return;

  float2 pos = particles.positions[g_id];

  float px_u = pos.x / cell_size;
  float py_u = (pos.y / cell_size) - 0.5f;
  int cx_u = max(0, min((int)px_u, (int)resolution.x - 2));
  int cy_u = max(0, min((int)py_u, (int)resolution.y - 2));
  float tx_u = px_u - cx_u;
  float ty_u = py_u - cy_u;

  float vel_u =
      (1 - tx_u) * (1 - ty_u) * grid_velocities[cx_u * resolution.y + cy_u].x +
      tx_u * (1 - ty_u) * grid_velocities[(cx_u + 1) * resolution.y + cy_u].x +
      tx_u * ty_u * grid_velocities[(cx_u + 1) * resolution.y + cy_u + 1].x +
      (1 - tx_u) * ty_u * grid_velocities[cx_u * resolution.y + cy_u + 1].x;

  float px_v = (pos.x / cell_size) - 0.5f;
  float py_v = pos.y / cell_size;
  int cx_v = max(0, min((int)px_v, (int)resolution.x - 2));
  int cy_v = max(0, min((int)py_v, (int)resolution.y - 2));
  float tx_v = px_v - cx_v;
  float ty_v = py_v - cy_v;

  float vel_v =
      (1 - tx_v) * (1 - ty_v) * grid_velocities[cx_v * resolution.y + cy_v].y +
      tx_v * (1 - ty_v) * grid_velocities[(cx_v + 1) * resolution.y + cy_v].y +
      tx_v * ty_v * grid_velocities[(cx_v + 1) * resolution.y + cy_v + 1].y +
      (1 - tx_v) * ty_v * grid_velocities[cx_v * resolution.y + cy_v + 1].y;

  particles.velocities[g_id] = make_float2(vel_u, vel_v);
}

void FlipFluid::init(std::string&& shaders_path) {
  cell_size = 0.02;
  particle_radius = 0.0033;
  s_textures =
      Shader(shaders_path + "/canvas.vert", shaders_path + "/canvas.frag");
  s_particles = Shader(shaders_path + "/particles.vert",
                       shaders_path + "/particles.frag");
  s_particles.use();
  s_particles.setMat4("u_projectionViewMatrix",
                      glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f));
  s_particles.setFloat("u_inPixelDiameter", particle_radius * 1000);

  resolution = make_uint2(size.x / cell_size, size.y / cell_size);
  particles_size = 12000;
  particle_boundings =
      make_float4(cell_size + 0.00001, size.x - cell_size - 0.00001,
                  cell_size + 0.00001, size.y - cell_size - 0.00001);
  mem_size = resolution.x * resolution.y;

  cudaMalloc(&d_busy_cells, sizeof(ushort2) * mem_size);
  cudaMalloc(&d_busy_cells_size, sizeof(unsigned int));

  cudaMalloc(&d_grid_velocities, sizeof(float2) * mem_size);
  cudaMalloc(&d_sum_of_weights, sizeof(float2) * mem_size);

  cudaMalloc(&d_grid, sizeof(float) * mem_size);
  cudaMalloc(&d_solid_cells, sizeof(float) * mem_size);

  block_size = dim3(32, 32);
  grid_size =
      dim3(((unsigned int)resolution.x + block_size.x - 1) / block_size.x,
           ((unsigned int)resolution.y + block_size.y - 1) / block_size.y);
  p_block_size = dim3(1024);
  p_grid_size = dim3((particles_size - 1) / p_block_size.x + 1);

  create_solid_cells<<<grid_size, block_size>>>(
      d_solid_cells, (unsigned int)resolution.x, (unsigned int)resolution.y);

  positions = std::vector<float2>(particles_size);
  velocities = std::vector<float2>(particles_size);

  std::uniform_real_distribution<float> rand_width(cell_size,
                                                   (size.x - cell_size) / 2.0);
  std::uniform_real_distribution<float> rand_height(cell_size,
                                                    size.y - cell_size);
  std::uniform_real_distribution<float> rand_velocity(-1, 1);
  for (int i = 0; i < particles_size; i++) {
    positions[i] = make_float2(rand_width(gen), rand_height(gen));
    velocities[i] = make_float2(rand_velocity(gen), rand_velocity(gen));
  }

  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, particles_size * sizeof(float2) * 2, nullptr,
               GL_DYNAMIC_DRAW);

  glBufferSubData(GL_ARRAY_BUFFER, 0, particles_size * sizeof(float2),
                  positions.data());
  glBufferSubData(GL_ARRAY_BUFFER, particles_size * sizeof(float2),
                  particles_size * sizeof(float2), velocities.data());

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), (void*)0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float2),
                        (void*)(particles_size * sizeof(float2)));

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO,
                               cudaGraphicsMapFlagsNone);

  collisions.setup(size.x, size.y, particles_size, particle_radius);

  simulate_particles_shared_size = sizeof(float2) * mem_size +
                                   sizeof(float) * mem_size +
                                   sizeof(unsigned short) * mem_size;
}

void FlipFluid::update() {
  size_t num_bytes;
  cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);

  float2* d_vbo_ptr;
  cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes,
                                       cuda_vbo_resource);

  d_particles.positions = d_vbo_ptr;
  d_particles.velocities = d_vbo_ptr + particles_size;

  for (int i = 0; i < 15; i++) {
    collisions.check_collision(d_particles.positions);
  }

  cudaMemset(d_busy_cells, 0, sizeof(ushort2) * mem_size);
  cudaMemset(d_busy_cells_size, 0, sizeof(unsigned int));

  cudaMemset(d_grid_velocities, 0, sizeof(float2) * mem_size);
  cudaMemset(d_sum_of_weights, 0, sizeof(float2) * mem_size);

  cudaMemset(d_grid, 0, sizeof(float) * mem_size);
  // std::cout << Time::frames_per_second << '\n';
  simulate_particles<<<p_grid_size, p_block_size>>>(
      d_grid, d_particles, d_grid_velocities, d_sum_of_weights, d_busy_cells,
      d_busy_cells_size, particles_size, resolution, mem_size,
      particle_boundings, cell_size, /*Time::delta_time*/ 0.004,
      interaction_data, is_interacting);
  is_interacting = false;
  update_velocities<<<grid_size, block_size>>>(d_grid_velocities,
                                               d_sum_of_weights, d_busy_cells,
                                               d_busy_cells_size, resolution);

  for (int i = 0; i < 40; i++) {
    calculate_divergence<<<grid_size, block_size>>>(
        d_grid, d_solid_cells, d_grid_velocities, resolution, i);
  }

  grid_to_particles<<<p_grid_size, p_block_size>>>(
      d_grid_velocities, d_particles, cell_size, particles_size, resolution);

  cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}

void FlipFluid::draw() {
  s_particles.use();
  glBindVertexArray(VAO);
  glDrawArrays(GL_POINTS, 0, particles_size);
  glBindVertexArray(0);
}

void FlipFluid::set_interaction_data(const glm::vec2& cursor_pos,
                                     const glm::vec2& cursor_vel) {
  this->interaction_data =
      make_float4(cursor_pos.x, cursor_pos.y, cursor_vel.x, cursor_vel.y);
  this->is_interacting = true;
}