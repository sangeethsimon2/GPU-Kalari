cmake_minimum_required(VERSION 3.23)

project(
    testSmartPtrs
    VERSION 0.1
)

#Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/app")

add_executable( testSmartPtrs )
 target_sources(testSmartPtrs PUBLIC
  src/main.cpp
  )

set(CMAKE_CXX_COMPILER "/softs/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin/nvc++")

find_package(OpenACC REQUIRED)
if(OpenACC_CXX_FOUND)
    message(STATUS "OpenACC found")
    set(CMAKE_CXX_FLAGS " -acc -gpu=cc70,managed -Minfo=all")
    #add_compile_options(-std=c++17 -acc -gpu=cc70,managed -Minfo=all)
    #target_compile_options(testSmartPtrs PUBLIC -acc -gpu=managed -Minfo=all)
    target_link_libraries(testSmartPtrs PUBLIC OpenACC::OpenACC_CXX)
endif()
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit found")
endif()






# set(CMAKE_CXX_COMPILER "/softs/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin/nvc++")
# set(CMAKE_CXX_FLAGS " -acc -gpu=cc70,managed -Minfo=all")

# #set(OpenACC_CXX_COMPILER "nvc++")
# #set(OpenACC_CXX_FLAGS " -acc -gpu=cc70,managed -Minfo=all")


# message(OPENACC_CXX_FLAGS= ${OpenACC_CXX_FLAGS})


# # Check if OpenACC::OpenACC_CXX is defined and print its value
# if(TARGET OpenACC_CXX)
#     get_target_property(OPENACC_CXX_LOCATION OpenACC_CXX IMPORTED_LOCATION)
#     message(STATUS "OpenACC CXX location: ${OPENACC_CXX_LOCATION}")
# else()
#     message(STATUS "OpenACC CXX target is not defined or found")
#endif()


# #message(OpenACC_CXX_VERSION_MAJOR= ${OpenACC_CXX_VERSION_MAJOR})
# message(CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}")
# message(CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}")

#  add_executable( testSmartPtrs )
# # target_include_directories(testSmartPtrs PUBLIC "${PROJECT_SOURCE_DIR}/include")
#  #target_compile_options(testSmartPtrs PUBLIC ${OPENACC_CXX_FLAGS})
#  #target_link_libraries(testSmartPtrs PUBLIC OpenACC::OpenACC_CXX)

#  target_compile_options(testSmartPtrs PUBLIC ${CMAKE_CXX_FLAGS})

#  target_sources(testSmartPtrs PRIVATE

#   src/main.cpp
#  )

