#include "Properties.hpp"

void Properties::load() {
  name = "flip-fluid-sim"; 
  scr_width = 800;
  scr_height = 800;
  fullscreen = false;
  background_color = glm::vec3(0.051f, 0.067f, 0.090f);
  frame_rate_limit = 0; 
}
