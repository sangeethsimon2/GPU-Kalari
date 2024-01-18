add_executable( matmult matmult.cpp)

option(ENABLE_CUDA "ENABLE_CUDA" ON)
if(ENABLE_CUDA)
  target_link_libraries( matmult PRIVATE  hostTimer gpuTimer MatrixBase MatMultOnGPUUsingCUDA)
  target_compile_definitions(matmult PRIVATE "ENABLE_CUDA")
endif()
