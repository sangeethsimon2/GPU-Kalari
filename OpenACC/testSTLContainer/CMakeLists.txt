cmake_minimum_required(VERSION 3.23)

project(
    testSTLContainer
    VERSION 0.1
    LANGUAGES CXX
    )

#Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/app")


# set(CMAKE_CUDA_COMPILER "/softs/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/bin")
# set(OpenACC_DIR "/softs/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers" CACHE PATH "Path to OpenACC installation")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenACC_DIR}/include")
#set(OpenACC_ACCEL_TARGET "cc70,managed,lineinfo")

find_package(OpenACC REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_COMPILER "nvc++")
# The 'managed' flag is important to achieve deep copy of vector container
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -gpu=cc70,managed -Minfo=all")

message(OPENACC_CXX_FLAGS= ${OpenACC_CXX_FLAGS})
message(OpenACC_CXX_VERSION_MAJOR= ${OpenACC_CXX_VERSION_MAJOR})
message(CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}")

add_executable( testSTLContainer )
# target_include_directories(testSTLContainer PUBLIC "${PROJECT_SOURCE_DIR}/include")
 target_compile_options(testSTLContainer PUBLIC ${OPENACC_CXX})
 target_link_libraries(testSTLContainer PUBLIC OpenACC::OpenACC_CXX)
 target_sources(testSTLContainer PRIVATE
                src/main.cpp
 )
