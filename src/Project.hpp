#pragma once

#include "Camera.hpp"
#include "FileLoaders.hpp"
#include "FrameHandler.hpp"
#include "Scene.hpp"
#include "Time.hpp"

class Project {
 public:
  Project();
  void init();
  void run();

  OrthographicCamera camera;
  Scene scene;
  Properties properties;

  operator GLFWwindow*() const { return window; };
  operator Properties() { return properties; };

 private:
  void frame_update();
  void frame_render();
  void frame_end();
  GLFWwindow* window;
};