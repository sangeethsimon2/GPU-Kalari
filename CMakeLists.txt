cmake_minimum_required(VERSION 3.23)

project(
    matrixMultiplication
    VERSION 0.1
    LANGUAGES CXX CUDA
    )

#Set CXX version and make it required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

option(ENABLE_SERIAL "ENABLE_SERIAL" OFF)
option(ENABLE_CUDA "ENABLE_CUDA" ON)

if (ENABLE_CUDA)
  find_package(CUDA  REQUIRED)
  include_directories("${CUDA_INCLUDE_DIRS}")
  set(CMAKE_CUDA_ARCHITECTURES "70;72")
  set(CMAKE_CUDA_STANDARD 17)
endif()

add_subdirectory(src)
add_subdirectory(app)

