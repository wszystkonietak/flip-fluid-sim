#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include "Callbacks.hpp"
#include "Project.hpp"

int main() {
  std::string path =
      "C:\\Users\\Igor\\Desktop\\CUDA_OpenGL_interop\\CUDA_OpenGL_"
      "interop\\wszystkonietak";
  Project project(path);
  Callbacks callbacks(project);
  project.run();
  return EXIT_SUCCESS;
}