#pragma once
#include <glm/glm.hpp>
#include <vector>

#include "DataTypes.hpp"
#include "Shader.hpp"

class Mesh {
 public:
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;
  std::vector<Texture> textures;

  Mesh() {}
  Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices,
       std::vector<Texture> textures);
  void setupMesh();
  void updateMesh();
  void draw(Shader& shader, GLenum mode = GL_TRIANGLES);

 protected:
  GLuint VAO, VBO, EBO;
};