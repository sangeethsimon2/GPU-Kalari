add_executable( matmult matmult.cpp)

if(ENABLE_SERIAL)
  target_compile_definitions(matmult PRIVATE "ENABLE_SERIAL")
endif()

if(ENABLE_CUDA)
  target_link_libraries( matmult PRIVATE  hostTimer gpuTimer MatrixBase MatMultOnGPUUsingCUDA)
  target_compile_definitions(matmult PRIVATE "ENABLE_CUDA")
endif()
