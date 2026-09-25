#include "ParticleSystem.cuh"

void ParticleSystem::init(float scene_width, float scene_height,
                          float particles_radius, int num_particles) {
  this->particle_radius = particles_radius;
  std::uniform_real_distribution<float> rand_height(
      4 * particles_radius, scene_height - 4 * particles_radius);
  std::uniform_real_distribution<float> rand_width(
      4 * particles_radius, scene_width - 4 * particles_radius);

  positions.reserve(num_particles);
  velocities.reserve(num_particles);
  for (int i = 0; i < num_particles; i++) {
    positions.push_back(make_float2(rand_width(gen), rand_height(gen)));
    velocities.push_back(make_float2(0, 0));
  }

  setupParticleSystem();

  cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO,
                               cudaGraphicsMapFlagsNone);

  collisions.setup(scene_width, scene_height, positions.size(),
                   particle_radius);
}

void ParticleSystem::draw(Shader& shader) {
  shader.use();
  glBindVertexArray(VAO);
  glDrawArrays(GL_POINTS, 0, positions.size());
  glBindVertexArray(0);
}

void ParticleSystem::update() {
  size_t num_bytes;

  cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);

  float2* d_vbo_ptr;
  cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes,
                                       cuda_vbo_resource);

  d_particles.positions = d_vbo_ptr;
  d_particles.velocities = d_vbo_ptr + positions.size();

  collisions.check_collision(d_particles.positions);

  cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}

void ParticleSystem::setupParticleSystem() {
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(
      GL_ARRAY_BUFFER,
      positions.size() * sizeof(float2) + velocities.size() * sizeof(float2),
      nullptr, GL_DYNAMIC_DRAW);

  glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(float2),
                  positions.data());
  glBufferSubData(GL_ARRAY_BUFFER, positions.size() * sizeof(float2),
                  velocities.size() * sizeof(float2), velocities.data());

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), (void*)0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float2),
                        (void*)(positions.size() * sizeof(float2)));

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindVertexArray(0);
}