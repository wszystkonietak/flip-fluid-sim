#pragma once
#include <vector>

#include "CollisionDetection.cuh"
#include "DataTypes.hpp"
#include "RandomSeed.hpp"
#include "Shader.hpp"
#include "Time.hpp"
#include "cuda_gl_interop.h"

class ParticleSystem {
 public:
  ParticleSystem() = default;
  ParticleSystem(float scene_width, float scene_height, float particle_radius,
                 int num_particles) {
    init(scene_width, scene_height, particle_radius, num_particles);
  }
  virtual void init(float scene_width, float scene_height,
                    float particles_radius, int num_particles);
  virtual void draw(Shader& shader);
  virtual void update();
  void setupParticleSystem();

  float particle_radius;
  CollisionDetection collisions;

  Particles d_particles;
  std::vector<float2> positions;
  std::vector<float2> velocities;

  unsigned int VAO;
  unsigned int VBO;

  cudaGraphicsResource* cuda_vbo_resource;
};