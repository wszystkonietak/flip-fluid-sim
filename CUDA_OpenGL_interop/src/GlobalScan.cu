#include <cuda_runtime.h>

#include "GlobalScan.cuh"

#define PAD(n) ((n) + ((n) >> 5))

__device__ void d_prefix_sum(unsigned int* values, unsigned int n,
                             unsigned int* block_sum_out) {
  int offset = 1;

  for (int d = n >> 1; d > 0; d >>= 1) {
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
      int ai = PAD(offset * (2 * i + 1) - 1);
      int bi = PAD(offset * (2 * i + 2) - 1);
      values[bi] += values[ai];
    }
    offset <<= 1;
  }

  if (!threadIdx.x) {
    int last_idx = PAD(n - 1);
    if (block_sum_out) *block_sum_out = values[last_idx];
    values[last_idx] = 0;
  }

  for (int d = 1; d < n; d *= 2) {
    offset >>= 1;
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
      int ai = PAD(offset * (2 * i + 1) - 1);
      int bi = PAD(offset * (2 * i + 2) - 1);
      unsigned int t = values[ai];
      values[ai] = values[bi];
      values[bi] += t;
    }
  }
}

__global__ void scanBlocksKernel(unsigned int* d_out, unsigned int* d_in,
                                 unsigned int* d_block_sums, unsigned int n,
                                 unsigned int epb) {
  extern __shared__ unsigned int temp[];

  int blockOffset = blockIdx.x * epb;
  int ai = threadIdx.x;
  int bi = threadIdx.x + blockDim.x;
  int g_ai = blockOffset + ai;
  int g_bi = blockOffset + bi;

  temp[PAD(ai)] = (g_ai < n) ? d_in[g_ai] : 0;
  temp[PAD(bi)] = (g_bi < n) ? d_in[g_bi] : 0;

  __syncthreads();

  unsigned int block_sum = 0;
  d_prefix_sum(temp, epb, &block_sum);

  __syncthreads();

  if (g_ai < n) d_out[g_ai] = temp[PAD(ai)];
  if (g_bi < n) d_out[g_bi] = temp[PAD(bi)];

  if (!threadIdx.x && d_block_sums) {
    d_block_sums[blockIdx.x] = block_sum;
  }
}

__global__ void addBlockSumsKernel(unsigned int* d_data,
                                   unsigned int* d_block_sums, unsigned int n,
                                   unsigned int epb) {
  int g_id = blockIdx.x * epb + threadIdx.x;
  unsigned int sum = d_block_sums[blockIdx.x];

  if (g_id < n) d_data[g_id] += sum;
  if (g_id + blockDim.x < n) d_data[g_id + blockDim.x] += sum;
}

void runGlobalPrefixSum(unsigned int* d_data, unsigned int n) {
  if (n == 0) return;

  unsigned int threads = 256;
  unsigned int epb = threads * 2;
  unsigned int num_blocks = (n + epb - 1) / epb;
  unsigned int shared_mem = PAD(epb) * sizeof(unsigned int);

  if (num_blocks <= 1) {
    scanBlocksKernel<<<1, threads, shared_mem>>>(d_data, d_data, nullptr, n,
                                                 epb);
    return;
  }

  unsigned int* d_block_sums;
  cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(unsigned int));

  scanBlocksKernel<<<num_blocks, threads, shared_mem>>>(d_data, d_data,
                                                        d_block_sums, n, epb);
  runGlobalPrefixSum(d_block_sums, num_blocks);
  addBlockSumsKernel<<<num_blocks, threads>>>(d_data, d_block_sums, n, epb);

  cudaFree(d_block_sums);
}