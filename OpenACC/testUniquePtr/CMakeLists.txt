cmake_minimum_required(VERSION 3.23)

project(
    testSmartPtrs
    VERSION 0.1
    LANGUAGES CXX
    )

#Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/app")

find_package(OpenACC REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_COMPILER "nvc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -gpu=cc70,managed -Minfo=all")

message(OPENACC_CXX_FLAGS= ${OpenACC_CXX_FLAGS})
message(OpenACC_CXX_VERSION_MAJOR= ${OpenACC_CXX_VERSION_MAJOR})
message(CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}")

 add_executable( testSmartPtrs )
# target_include_directories(testSmartPtrs PUBLIC "${PROJECT_SOURCE_DIR}/include")
 target_compile_options(testSmartPtrs PUBLIC ${OPENACC_CXX})
 target_link_libraries(testSmartPtrs PUBLIC OpenACC::OpenACC_CXX)
 target_sources(testSmartPtrs PRIVATE
                src/main.cpp
 )

 # set(CMAKE_CUDA_COMPILER "/softs/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/bin")
# set(OpenACC_DIR "/softs/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers" CACHE PATH "Path to OpenACC installation")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenACC_DIR}/include")
#set(OpenACC_ACCEL_TARGET "cc70,managed,lineinfo")


# set(OpenACC_LINKFLAGS "-acc")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"--std=c++17\"")

# message(CMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS}")
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     message(STATUS "Enabling -O0 optimization flag")
#     add_compile_options(-O0 -g)
# endif()

# if (CMAKE_BUILD_TYPE STREQUAL "Release")
#     set(CMAKE_CUDA_ARCHITECTURES "70;72")
#     set(CMAKE_CUDA_STANDARD 17)
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"--std=c++17\"")
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ta=cc70 -Minfo=all,intensity  ")
# #    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -fast -Minfo=all,intensity -fopt-info-optimized=report.opt -O3 -funroll-loops -march=native ")
# endif()