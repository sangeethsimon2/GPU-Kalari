#include<iostream>
#include "AABB.h"
#include "smartBuffer.cuh"

int main() {
  
    //Can be user inputs
    const size_t numObjs = 10;
    int threadsPerBlock = 256;
   
    //Create data on host
    auto rects = createObjOnScreen(numObjs, objType::RECTANGLE);
    
    //Sample container to hold downloaded data from device 
    std::vector<rectangleObject> returnedRects; 
    returnedRects.reserve(numObjs);

    std::vector<AABB> returnedAABBs;
    returnedAABBs.reserve(numObjs);

    //Copy data to device
    //uploadTodevice(rects.data(), numObjs); 
  
    //Kernel launch parameters
    int blocksPerGrid = (numObjs + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "Launching CUDA kernel with " << blocksPerGrid << " blocks and "
              << threadsPerBlock << " threads per block..." << std::endl;

    //computeTree(numObjs, blocksPerGrid, threadsPerBlock);
    
    //Copy back from device to host'
    //testdownloadToHost(returnedRects.data(), numObjs); 

    //downloadToHost(returnedAABBs.data(), numObjs);

    SmartMemoryManager::HostDeviceMemoryManager<rectangleObject> rect_smartBuffer(rects.data(), numObjs);    
    rect_smartBuffer.uploadToDevice();
    rect_smartBuffer.downloadToHost();
    SmartMemoryManager::HostDeviceMemoryManager<AABB> AABB_smartBuffer(returnedAABBs.data(), numObjs);
    AABB_smartBuffer.uploadToDevice();
    AABB_smartBuffer.downloadToHost();
    return 0;
}