#include<iostream>
#include"bvh.h"
int main() {
    // 1. Define the number of rectangular objects
    const int num_rects = 10;
     int threadsPerBlock = 256;
    // Calculate the number of blocks needed to cover all rectangles
    int blocksPerGrid = (num_rects + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching CUDA kernel with " << blocksPerGrid << " blocks and "
              << threadsPerBlock << " threads per block..." << std::endl;

    computeTree(num_rects, blocksPerGrid, threadsPerBlock);


    return 0;
}