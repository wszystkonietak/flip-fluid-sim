#pragma once
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Camera.hpp"
#include "Canvas.cuh"
#include "ParticleSystem.cuh"
#include "Properties.hpp"
#include "Shader.hpp"
#include "SoftBody.cuh"
#include "Water.cuh"

class Scene {
 public:
  Scene() {}
  Scene(std::vector<SoftBody> softBodies, std::vector<Shader> shaders);
  void load();
  void render();
  void setCameraZoom(const OrthographicCamera& camera, FrameHandler& input);
  void setCameraProjection(const OrthographicCamera& camera);
  void updateShaders(OrthographicCamera& camera);
  void userInteraction(FrameHandler& frame);
  void updateMeshes();

 private:
  void loadShaders();
  void loadCanvases();
  void loadFluids();
  std::string scene_path;
  std::vector<FlipFluid> fluids;
  std::vector<SoftBody> softBodies;
  std::vector<Canvas> canvases;
  std::vector<ParticleSystem> particles;
  std::vector<Shader> shaders;
};