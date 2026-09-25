#pragma once
#include <stdio.h>

#include <glm/gtc/matrix_transform.hpp>
#include <vector>

#include "CollisionDetection.cuh"
#include "RandomSeed.hpp"
#include "Shader.hpp"
#include "Time.hpp"

class FlipFluid {
 public:
  FlipFluid(glm::vec2 size)
      : size(size) {
    init();
  }
  void init();
  void update();
  void draw();
  void set_interaction_data(const glm::vec2& cursor_pos,
                            const glm::vec2& cursor_vel);
  float4 interaction_data;
  bool is_interacting = false;
  float4 particle_boundings;
  float cell_size;
  float particle_radius;
  unsigned int particles_size;
  unsigned int rest_particle_density = 500;
  unsigned int num_iters = 2;
  uint2 resolution;
  glm::vec2 size;

  std::vector<float2> positions;
  std::vector<float2> velocities;
  CollisionDetection collisions;
  unsigned int mem_size, simulate_particles_shared_size;

  float* d_solid_cells;
  float* d_grid;
  cudaGraphicsResource* cuda_vbo_resource;
  Particles d_particles;
  ushort2* d_busy_cells;
  float2* d_grid_velocities;
  float2* d_sum_of_weights;
  unsigned int* d_busy_cells_size;
  unsigned int VAO, VBO;
  Shader s_textures, s_particles;
  dim3 block_size;
  dim3 grid_size;
  dim3 p_block_size;
  dim3 p_grid_size;
};