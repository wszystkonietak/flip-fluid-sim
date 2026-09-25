#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "DataTypes.hpp"
#include "GlobalScan.cuh"
#include "RadixSort.cuh"

class CollisionDetection {
 public:
  CollisionDetection() = default;
  ~CollisionDetection();

  void setup(float scene_width, float scene_height, unsigned int size,
             float radius);

  void check_collision(float2* d_positions);


  unsigned int get_gpu_collisions_count() const;
  void get_sorted_data(std::vector<unsigned int>& h_cells,
                       std::vector<unsigned int>& h_objects,
                       unsigned int& h_cell_count) const;

  float get_diameter() const { return diameter; }
  float get_radius() const { return radius; }
  unsigned int get_x_shift() const { return x_shift; }

 private:
  unsigned int count_bits(unsigned int n);

  unsigned int objects_size;
  float scene_width;
  float scene_height;
  float radius;
  float diameter;

  unsigned int width;
  unsigned int height;
  unsigned int x_shift;
  unsigned int min_bits_for_hash;
  unsigned int max_grid_capacity;
  unsigned int cells_size;
  float max_distance_squared;

  unsigned int num_blocks;
  unsigned int num_threads;
  unsigned int shared_size;

  unsigned int* d_cells = nullptr;
  unsigned int* d_objects = nullptr;
  unsigned int* d_cell_count = nullptr;
  unsigned int* d_control_bits = nullptr;

  uint3* d_collision_cells_uncompacted = nullptr;
  unsigned int* d_collision_cells_ids = nullptr;
  unsigned int* d_collisions_count = nullptr;
  uint3* d_collision_cells = nullptr;

  unsigned int h_cell_count = 0;
  unsigned int h_gpu_collisions = 0;

  RadixSort sorter;
};