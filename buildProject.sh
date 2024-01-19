module load tools/cmake/3.23.0
module load nvidia/cuda/12.0
rm -rf CMakeFiles
rm  Makefile
rm CMakeCache.txt
rm cmake_install.cmake
cmake -DENABLE_CUDA=ON -DENABLE_SERIAL=OFF ../CMakeLists.txt
make
