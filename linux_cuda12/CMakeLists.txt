cmake_minimum_required(VERSION 3.18)
project(PainlessMG_Linux LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - adjust based on your GPU
# Common options: 60, 61, 70, 75, 80, 86, 89, 90
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")

# Find required packages
find_package(CUDA REQUIRED)

# OpenGL - use modern approach with components
find_package(OpenGL REQUIRED COMPONENTS OpenGL GLX)
if(NOT OpenGL_FOUND)
    # Fallback: try to find libraries manually
    find_library(OPENGL_opengl_LIBRARY NAMES OpenGL GL)
    find_library(OPENGL_glx_LIBRARY NAMES GLX)
    if(OPENGL_opengl_LIBRARY AND OPENGL_glx_LIBRARY)
        set(OpenGL_FOUND TRUE)
        set(OPENGL_LIBRARIES ${OPENGL_opengl_LIBRARY} ${OPENGL_glx_LIBRARY})
    else()
        message(FATAL_ERROR "OpenGL not found. Please install: sudo apt-get install libopengl-dev libglx-dev")
    endif()
endif()

find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/lib_fixed
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_lib/eigen
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)

# Common CUDA flags for CUDA 12
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# Add subdirectories
add_subdirectory(armadillo)
add_subdirectory(cloth)
