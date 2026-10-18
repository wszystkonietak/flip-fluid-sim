# Real-Time FLIP Fluid Simulation (CUDA + OpenGL Interop)

A 2D FLIP fluid simulation written in C++ and CUDA rendered using OpenGL. The fluid solver is based on Matthias Müller's [Ten Minute Physics FLIP tutorial](https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/18-flip.html), ported to run on the GPU. Broad-phase particles collision detection is based on [GPU Gems 3 (Chapter 32)](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda).

The simulation runs on the GPU with no data copies between the CPU and GPU during the simulation loop. Particle data is written directly into an OpenGL Vertex Buffer Object (VBO).

https://github.com/user-attachments/assets/54c06561-f404-48c5-af3e-17ef7c0e9dc4

## Pipeline Architecture

Each frame runs these steps:

1. **Advection & Particle to Grid**: Particles move with gravity and mouse drag. Velocities transfer onto the staggered grid and are normalized by cell weights.
2. **Pressure Solve**: A Red-Black Gauss-Seidel solver runs across grid cells to keep the fluid incompressible.
3. **Grid to Particle**: New grid velocities are sampled back to the particles using a FLIP update.
4. **Collision Detection**: Collisions follow the *GPU Gems 3* broad-phase spatial hashing approach:
   - **Home & Phantom Cells**: Particles get a home cell, plus phantom cells if they cross cell boundaries.
   - **Radix Sort**: A 4-pass 8-bit Radix Sort groups particle entries by cell hash.
   - **Compaction**: Active cells are compacted into a contiguous array using a prefix scan.
   - **Resolution**: Cells run across 4 passes to prevent write conflicts. Overlapping particles in each cell are pushed apart.
## Kernel Bottleneck Analysis

Profiling the collision pipeline with 2M particles and 5.2M collisions (30.2 ms on an NVIDIA RTX 3060) shows where execution time is spent:

- `radixReorder` (4 passes) — 15.6 ms (52%): Moves spatial cell and particle IDs into their sorted positions during each radix sort pass.
- `resolveCollisions` (4 passes) — 9.1 ms (30%): Resolves particle-to-particle collisions, with execution time mainly limited by global memory read latency.
- `countCollisionCells` (1 pass) — 3.6 ms (12%): Scans sorted cell IDs to find cells with colliding particles. Execution time is mainly affected by strided global memory reads (`CollisionDetection.cu:273, 298`) as each thread loops over a separate segment of the array.
- Setup, Scan & Compaction Kernels (16 launches) — 1.9 ms (6%): Handles data initialization, prefix-sum scans, stream compaction, and boundary conditions.

### Why `radixReorder` is slow?

In each pass of the radix sort, `radixReorder` reads the unsorted `cells` and `objects` arrays, extracts the current 8-bit radix digit for every element, finds its destination index using prefix-sum offsets in shared memory, and writes both values to `cells_out` and `objects_out`. Because `radixReorder` takes over half of the execution budget, overall simulation speed depends on its memory access pattern.

In `RadixSort.cu`, execution time is concentrated in the final reordering loop (lines 198–208) due to three hardware behaviors:

1. Global Memory Read Stalls (`RadixSort.cu:201, 207`):
   ```cuda
   unsigned int digit = (cells[i] >> shift) & (constants.num_radices - 1);
   ```
   In each loop iteration, threads load unsorted cells[i] (line 201) and objects[i] (line 207) from global memory. Although these reads are mostly contiguous within each thread group, they compete for L2 cache and DRAM bandwidth with the scattered writes at the end of the loop. Because of this high memory traffic, warps spend 41% of their stall time at line 201 waiting for cells[i] to arrive from memory (and also stall at line 207 waiting for objects[i] to load before writing it out).

2. Shared Memory Atomic Serialization (`RadixSort.cu:204`):
   ```cuda
   unsigned int target_idx = atomicAdd(&t[index], 1);
   ```
   To give each element a unique output index inside its bucket, threads increment a shared memory counter `t[index]`. Because nearby particles often share the same 8-bit radix digit during a pass, multiple threads in the same warp frequently call `atomicAdd` on the exact same shared memory address at the same time. These conflicting accesses are processed sequentially, making up 80% of all shared memory traffic in the kernel and forcing threads to wait at line 206 until `target_idx` is ready.

3. Scattered Global Writes (`RadixSort.cu:206–207` — 55% of stalls):
   ```cuda
   cells_out[target_idx] = cells[i];
   objects_out[target_idx] = objects[i];
   ```
   After getting `target_idx`, each thread writes its cell ID and object ID to the output arrays. Whenever threads in the same warp process elements that belong to different radix buckets, their `target_idx` addresses point to distant places in global memory. Because the GPU accesses global memory in 32-byte sectors, writing to non-adjacent addresses uses only 5.5 of 32 bytes per sector on average. This creates 92% of the kernel's wasted global memory traffic and bottlenecks the L2 cache.

## Dependencies

* C++17 compiler
* CMake (>= 3.18)
* CUDA Toolkit (>= 11.0)
* OpenGL, GLFW, GLM, Glad

## References

* **Matthias Müller** — [Ten Minute Physics: 18 - FLIP Water](https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/18-flip.html)
* **Scott Le Grand** — [GPU Gems 3, Chapter 32: Broad-Phase Collision Detection with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)
