#include "Scene.hpp"

Scene::Scene(std::vector<SoftBody> softBodies, std::vector<Shader> shaders) {
  this->softBodies = softBodies;
  this->shaders = shaders;
}

void Scene::load() {
  loadShaders();
  loadCanvases();
  loadFluids();
}

void Scene::updateShaders(OrthographicCamera& camera) {
  shaders[s_Basic].use();
  shaders[s_Basic].setMat4("u_projectionViewMatrix",
                           camera.getProjectionViewMatrix());
  shaders[s_SoftBody].use();
  shaders[s_SoftBody].setMat4("u_projectionViewMatrix",
                              camera.getProjectionViewMatrix());
  shaders[s_SoftBody].use();
  shaders[s_SoftBody].setMat4("u_projectionViewMatrix",
                              camera.getProjectionViewMatrix());
}

void Scene::userInteraction(FrameHandler& frame) {
  glm::vec2 scr_sz = {frame.properties.scr_width, frame.properties.scr_height};
  for (auto& fluid : fluids) {
    fluid.set_interaction_data(frame.mouse.cursor.current,
                               frame.mouse.cursor.velocity);
  }
}

void Scene::updateMeshes() {
  for (auto& softBody : softBodies) {
    softBody.simulate();
  }
  for (auto& particleSystem : particles) {
    particleSystem.update();
  }
  for (auto& canvas : canvases) {
    canvas.update();
  }
  for (auto& fluid : fluids) {
    fluid.update();
  }
}

void Scene::render() {
  shaders[s_Basic].use();
  for (auto& softBody : softBodies) {
    softBody.draw(shaders[s_SoftBody], points);
  }
  for (auto& particleSystem : particles) {
    particleSystem.draw(shaders[s_Particles]);
  }
  for (auto& canvas : canvases) {
    canvas.draw();
  }
  for (auto& fluid : fluids) {
    fluid.draw();
  }
}

void Scene::setCameraZoom(const OrthographicCamera& camera,
                          FrameHandler& input) {
  shaders[s_Basic].use();
  shaders[s_Basic].setMat4("u_projectionViewMatrix",
                           camera.getProjectionViewMatrix());
  float inPixelDiameter = 0;
  if (softBodies.size() > 0) {
    inPixelDiameter =
        2 * softBodies[0].particle_radius * camera.getFrustumRatio();
    shaders[s_Basic].setFloat("u_inPixelDiameter", inPixelDiameter);
  }

  shaders[s_SoftBody].use();
  shaders[s_SoftBody].setMat4("u_projectionViewMatrix",
                              camera.getProjectionViewMatrix());
  if (softBodies.size() > 0) {
    shaders[s_SoftBody].setFloat("u_inPixelDiameter", inPixelDiameter);
  }

  shaders[s_Particles].use();
  shaders[s_Particles].setMat4("u_projectionViewMatrix",
                               camera.getProjectionViewMatrix());
  if (particles.size() > 0) {
    float inPixelRadius = particles[0].particle_radius *
                          (input.properties.scr_width / camera.getFrustum().y);
    inPixelDiameter = 2 * inPixelRadius;
    shaders[s_Particles].setFloat("u_inPixelRadius", inPixelRadius);
    shaders[s_Particles].setFloat("u_inPixelDiameter", inPixelDiameter);
  }
}

void Scene::setCameraProjection(const OrthographicCamera& camera) {
  shaders[s_Basic].use();
  shaders[s_Basic].setMat4("u_projectionViewMatrix",
                           camera.getProjectionViewMatrix());

  shaders[s_SoftBody].use();
  shaders[s_SoftBody].setMat4("u_projectionViewMatrix",
                              camera.getProjectionViewMatrix());

  shaders[s_Particles].use();
  shaders[s_Particles].setMat4("u_projectionViewMatrix",
                               camera.getProjectionViewMatrix());
}

void Scene::loadShaders() {
  const std::string shaderDir = "shaders/";

  const std::vector<std::string> shaderNames = {"basic", "softBody",
                                                "particles"};

  shaders.clear();
  shaders.reserve(shaderNames.size());

  for (const auto& name : shaderNames) {
    std::string vertPath = shaderDir + name + ".vert";
    std::string fragPath = shaderDir + name + ".frag";
    std::string geomPath = shaderDir + name + ".geom";

    std::ifstream geomFile(geomPath);
    if (geomFile.good()) {
      shaders.push_back(Shader(vertPath, fragPath, geomPath));
    } else {
      shaders.push_back(Shader(vertPath, fragPath, ""));
    }
  }
}

void Scene::loadCanvases() {
  // canvases.push_back(Canvas(glm::vec4(-1, -1, 1, 1), glm::vec2(1000, 1000),
  // scene_path + "/Shaders/"));
}

void Scene::loadFluids() {
  fluids.emplace_back(glm::vec2(1.0f, 1.0f));
}
