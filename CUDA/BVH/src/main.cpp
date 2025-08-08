#include<iostream>
#include "AABB.h"

int main() {
  
    //Can be user inputs
    const size_t numObjs = 10;
    int threadsPerBlock = 256;
   
    //Create data on host
    auto rects = createObjOnScreen(numObjs, objType::RECTANGLE);
    
    //Sample container to hold downloaded data from device 
    std::vector<rectangleObject> returnedRects; 
    returnedRects.reserve(numObjs);

    //Copy data to device
    uploadTodevice(rects.data(), numObjs); 
  
    //Kernel launch parameters
    int blocksPerGrid = (numObjs + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Launching CUDA kernel with " << blocksPerGrid << " blocks and "
              << threadsPerBlock << " threads per block..." << std::endl;

    computeTree(numObjs, blocksPerGrid, threadsPerBlock);
    
    //Copy back from device to host'
    testdownloadToHost(returnedRects.data(), numObjs); 
 

    return 0;
}