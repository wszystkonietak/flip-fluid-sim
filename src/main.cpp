#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include "Callbacks.hpp"
#include "Project.hpp"

int main() {
  Project project;
  Callbacks callbacks(project);
  project.run();
  return 0;
}