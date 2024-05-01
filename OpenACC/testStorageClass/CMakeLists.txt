cmake_minimum_required(VERSION 3.23)

project(
    testStorageClass
    VERSION 0.1
    LANGUAGES CXX
    )

#Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/app")

find_package(OpenACC REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_COMPILER "nvc++")
# The 'managed' flag is important to achieve deep copy of vector container
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -gpu=cc70,managed -Minfo=all")

message(OPENACC_CXX_FLAGS= ${OpenACC_CXX_FLAGS})
message(OpenACC_CXX_VERSION_MAJOR= ${OpenACC_CXX_VERSION_MAJOR})
message(CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}")

add_executable( testStorageClass )
 target_include_directories(testStorageClass PUBLIC "${PROJECT_SOURCE_DIR}/include")
 target_compile_options(testStorageClass PUBLIC ${OPENACC_CXX})
 target_link_libraries(testStorageClass PUBLIC OpenACC::OpenACC_CXX)
 target_sources(testStorageClass PRIVATE
                src/main.cpp
 )
