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

## Dependencies

* C++17 compiler
* CMake (>= 3.18)
* CUDA Toolkit (>= 11.0)
* OpenGL, GLFW, GLM, Glad

## References

* **Matthias Müller** — [Ten Minute Physics: 18 - FLIP Water](https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/18-flip.html)
* **Scott Le Grand** — [GPU Gems 3, Chapter 32: Broad-Phase Collision Detection with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)