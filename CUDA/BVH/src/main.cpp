#include<iostream>
#include "AABB.h"

int main() {
  
    //Can be user inputs
    const size_t num_rects = 10;
    int threadsPerBlock = 256;
   
    //Create data on host
    auto rects = createObjOnScreen(num_rects, objType::RECTANGLE);
    
    //Sample container to hold downloaded data from device 
    std::vector<rectangleObject> returnedRects; 
    returnedRects.reserve(num_rects);

    //Copy data to device
    uploadTodevice(rects.data(), num_rects); 

    //Copy back from device to host'
    testdownloadToHost(returnedRects.data(), num_rects); 
    
    int blocksPerGrid = (num_rects + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Launching CUDA kernel with " << blocksPerGrid << " blocks and "
              << threadsPerBlock << " threads per block..." << std::endl;

    //computeTree(num_rects, blocksPerGrid, threadsPerBlock);


    return 0;
}