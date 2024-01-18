include(CMakePrintHelpers)
message(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
#cmake_print_variables(PROJECT_SOURCE_DIR)

option(ENABLE_CUDA "ENABLE_CUDA" ON)

set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/hostTimer.hpp")
add_library(hostTimer hostTimer.cpp ${HEADER_LIST})
target_include_directories(hostTimer PUBLIC "${PROJECT_SOURCE_DIR}/include")


set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/MatrixBase.hpp")
add_library(MatrixBase MatrixBase.cpp ${HEADER_LIST})
target_include_directories(MatrixBase PUBLIC "${PROJECT_SOURCE_DIR}/include")

if (ENABLE_CUDA)
  set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/gpuTimer.hpp")
  add_library(gpuTimer gpuTimer.cpp ${HEADER_LIST})
  target_include_directories(gpuTimer PUBLIC "${PROJECT_SOURCE_DIR}/include")

  set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/MatMultOnGPUUsingCUDA.cuh")
  add_library(MatMultOnGPUUsingCUDA MatMultOnGPUUsingCUDA.cu ${HEADER_LIST})
  target_include_directories(MatMultOnGPUUsingCUDA PUBLIC "${PROJECT_SOURCE_DIR}/include")
endif()