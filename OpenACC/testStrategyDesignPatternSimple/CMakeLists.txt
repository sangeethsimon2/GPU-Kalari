cmake_minimum_required(VERSION 3.23)

project(
    testStrategyDesignPatternSimple
    VERSION 0.1
)

#Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/app")

add_executable( testStrategyDesignPatternSimple )
 target_sources(testStrategyDesignPatternSimple PUBLIC
  src/main.cpp
  )
target_include_directories(testStrategyDesignPatternSimple PUBLIC "${PROJECT_SOURCE_DIR}/include")

set(CMAKE_CXX_COMPILER "/softs/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin/nvc++")

find_package(OpenACC REQUIRED)
if(OpenACC_CXX_FOUND)
    message(STATUS "OpenACC found")
    set(CMAKE_CXX_FLAGS " -acc -gpu=cc70,managed -Minfo=all")
    #add_compile_options(-std=c++17 -acc -gpu=cc70,managed -Minfo=all)
    #target_compile_options(testStrategyDesignPatternSimple PUBLIC -acc -gpu=managed -Minfo=all)
    target_link_libraries(testStrategyDesignPatternSimple PUBLIC OpenACC::OpenACC_CXX)
endif()
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit found")
endif()



