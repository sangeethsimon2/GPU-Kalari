#!/bin/bash
# To be executed from a build/ directory
rm -rf CMakeFiles
rm  Makefile
rm CMakeCache.txt
rm cmake_install.cmake
#cmake -DENABLE_CUDA=ON -DENABLE_SERIAL=OFF --trace-expand > trace.txt 2>&1 ../CMakeLists.txt

cmake -DENABLE_CUDA=ON -DENABLE_SERIAL=OFF ../CMakeLists.txt
make
