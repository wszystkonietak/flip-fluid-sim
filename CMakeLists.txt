cmake_minimum_required(VERSION 3.18)

project(CUDA_OpenGL_interop LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(glad STATIC vendor/glad/src/glad.c)
target_include_directories(glad PUBLIC vendor/glad/include)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/glfw)

file(GLOB_RECURSE APP_SOURCES 
    "CUDA_OpenGL_interop/src/*.cpp"
    "CUDA_OpenGL_interop/src/*.h"
    "CUDA_OpenGL_interop/src/*.hpp"
    "CUDA_OpenGL_interop/src/*.cu"
    "CUDA_OpenGL_interop/src/*.cuh"
)

add_executable(CUDA_OpenGL_interop ${APP_SOURCES})

target_include_directories(CUDA_OpenGL_interop PRIVATE
    vendor/glm
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(CUDA_OpenGL_interop PRIVATE 
    GLFW_INCLUDE_NONE
)

target_link_libraries(CUDA_OpenGL_interop PRIVATE 
    glad 
    glfw 
    CUDA::cudart
)

set_target_properties(CUDA_OpenGL_interop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(CUDA_OpenGL_interop PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math 
        --ptxas-options=-v 
        -maxrregcount=32
    >
)

# ------------------------------------------------------------------------------
# Code Analysis & Formatting (Google Style)
# ------------------------------------------------------------------------------
# Required by clang-tidy to understand project include paths and compiler flags
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CLANG_FORMAT_EXE clang-format)
find_program(CLANG_TIDY_EXE clang-tidy)

# Clang-Format: Run manually via `cmake --build build --target format`
if(CLANG_FORMAT_EXE)
    add_custom_target(
        format
        COMMAND ${CLANG_FORMAT_EXE} -i -style=Google ${APP_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Applying Google style with clang-format..."
    )
else()
    message(WARNING "clang-format not found. 'format' target will not be available.")
endif()