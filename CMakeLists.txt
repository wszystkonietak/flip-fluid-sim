cmake_minimum_required(VERSION 3.18)

project(flip-fluid-sim LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
# Glad
add_library(glad STATIC vendor/glad/src/glad.c)
target_include_directories(glad PUBLIC vendor/glad/include)

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/glfw)

# ------------------------------------------------------------------------------
# Target & Sources
# ------------------------------------------------------------------------------
file(GLOB_RECURSE APP_SOURCES 
    "src/*.cpp"
    "src/*.hpp"
    "src/*.cu"
    "src/*.cuh"
)

add_executable(flip-fluid-sim ${APP_SOURCES})

target_include_directories(flip-fluid-sim PRIVATE
    src
    vendor/glad/include
    vendor/glm
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(flip-fluid-sim PRIVATE 
    GLFW_INCLUDE_NONE
)

target_link_libraries(flip-fluid-sim PRIVATE 
    glad 
    glfw 
    CUDA::cudart
)

set_target_properties(flip-fluid-sim PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(flip-fluid-sim PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math 
        --ptxas-options=-v 
        -maxrregcount=32
    >
)

# ------------------------------------------------------------------------------
# Post-Build: Copy Shaders to Output Directory
# ------------------------------------------------------------------------------
# Automatically copies shaders alongside the binary so relative paths never break
add_custom_command(TARGET flip-fluid-sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:flip-fluid-sim>/shaders
)