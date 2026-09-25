# Real-Time Fluid Simulation (CUDA + OpenGL Interop)

https://github.com/user-attachments/assets/54c06561-f404-48c5-af3e-17ef7c0e9dc4

A real-time, GPU-accelerated 2D FLIP fluid simulation written in C++ and CUDA with direct OpenGL interop. The physics and grid transfers are based on Matthias Müller's [Ten Minute Physics FLIP tutorial](https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/18-flip.html), fully ported to a parallel CUDA pipeline. Broad-phase particle collisions are handled via spatial hashing based on [GPU Gems 3 (Chapter 32)](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda), enhanced with atomic operations.

The simulation runs at **60+ FPS with 50,000 particles** (performing 20 collision substeps per frame) on an NVIDIA RTX 3060.

## Pipeline Architecture

Each frame executes entirely on the GPU:

1. **Advection & Particle to Grid Transfer:** Particle positions update with gravity; particle velocities are scattered onto a staggered MAC grid via bilinear interpolation.
2. **Pressure Solve (Incompressibility):** A parallel **Red-Black Gauss-Seidel / SOR** solver runs across 40 iterations on the grid to eliminate velocity divergence. The checkerboard pattern `(x + y) % 2 == (iteration % 2)` prevents race conditions between adjacent cells.
3. **Grid to Particles Transfer:** Updated, divergence-free grid velocities are interpolated back to particle velocities using FLIP blending.
4. **Broad-Phase Collision Detection:**
   * Particles are hashed into grid cells.
   * A custom 8-bit Radix Sort with a Blelloch prefix scan orders cells and particles.
   * Collisions are resolved in 4 independent spatial passes, allowing fully parallel processing without memory write hazards.
5. **Zero-Copy Rendering:** CUDA maps directly to the registered OpenGL Vertex Buffer Object (VBO) via `cudaGraphicsMapResources`. Particles render as point primitives without CPU staging.

## Optimizations & Profiling

Profiled and validated using **NVIDIA Nsight Compute**:

* **Conflict-Free Radix Sort:** Refactored the histogram setup kernel using a memory padding offset to successfully eliminate 32-way shared memory bank conflicts.
* **Zero Host-Device Transfers:** Grid velocities, pressure data, and mapped VBO positions remain in device memory during active simulation loops.

## Controls

* **Left Click + Drag:** Apply directional force and inject momentum into fluid particles.
* **Escape:** Close simulation.

## Build Instructions

### Prerequisites
* CMake 3.18+
* CUDA Toolkit 11.0+
* C++17 compatible compiler
* GLFW, GLM, Glad (configured in repository)

```bash
git clone --recursive https://github.com/wszystkonietak/flip-fluid-sim.git
cd flip-fluid-sim
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## References

* **Matthias Müller** — [Ten Minute Physics: 18 - FLIP Water](https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/18-flip.html)
* **Scott Le Grand** — [GPU Gems 3, Chapter 32: Broad-Phase Collision Detection with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)
