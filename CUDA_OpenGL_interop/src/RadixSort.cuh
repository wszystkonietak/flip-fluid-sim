#pragma once
#include <cstddef>

struct ConstantsRadixSetUp {
  unsigned int groups_per_block;
  unsigned int threads_per_group;
  unsigned int shared_memory_size;
  unsigned int cells_per_group;
  unsigned int num_radices;
  unsigned int cells_size;
};

struct ConstantsRadixSum {
  unsigned int radices_per_block;
  unsigned int num_radices;
  unsigned int num_groups;
  unsigned int padded_groups;
};

struct ConstantsRadixReorder {
  unsigned int groups_per_block;
  unsigned int threads_per_group;
  unsigned int num_radices;
  unsigned int cells_per_group;
  unsigned int radices_per_block;
  unsigned int padded_blocks;
  unsigned int cells_size;
  unsigned int sum_blocks;
};

class RadixSort {
 private:
  unsigned int* d_cells;
  unsigned int* d_objects;

  unsigned int* d_cells_tmp;
  unsigned int* d_objects_tmp;
  unsigned int* d_radices;
  unsigned int* d_radices_prefix_sum;

  unsigned int array_size;
  unsigned int min_bits_for_hash;
  unsigned int num_radices;
  unsigned int threads_per_group;

  int num_blocks_main;
  int num_threads_main;
  int num_blocks_sum;
  int num_threads_sum;

  size_t setup_smem_bytes;
  size_t sum_smem_bytes;
  size_t reorder_smem_bytes;

  ConstantsRadixSetUp constantsRadixSetUp;
  ConstantsRadixSum constantsRadixSum;
  ConstantsRadixReorder constantsRadixReorder;

  unsigned int succesive_power_of_two(unsigned int v);

 public:
  RadixSort();
  ~RadixSort();

  void setup(unsigned int array_size, unsigned int min_bits_for_hash,
             unsigned int* d_cells, unsigned int* d_objects);
  void sort();
};