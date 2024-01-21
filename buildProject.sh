module load tools/cmake/3.23.0
module load nvidia/cuda/11.4
rm -rf CMakeFiles
rm  Makefile
rm CMakeCache.txt
rm cmake_install.cmake
#cmake -DENABLE_CUDA=ON -DENABLE_SERIAL=OFF --trace-expand > trace.txt 2>&1 ../CMakeLists.txt

cmake -DENABLE_CUDA=ON -DENABLE_SERIAL=OFF ../CMakeLists.txt
make
