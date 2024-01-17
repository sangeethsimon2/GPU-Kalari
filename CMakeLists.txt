cmake_minimum_required(VERSION 3.15)

project(
    matrixMultiplication
    VERSION 0.1
    LANGUAGES CXX CUDA
    )

#Set CXX version and make it required
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_subdirectory(src)
add_subdirectory(app)



