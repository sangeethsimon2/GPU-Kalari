add_executable( matmult matmult.cpp)
target_link_libraries( matmult PRIVATE  hostTimer gpuTimer MatrixBase MatMultOnGPUUsingCUDA)