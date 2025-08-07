#include "AABB.h"
#include "bvh.cuh"
#include "mycudaUtilities.cuh"
#include <cuda_runtime.h>

//Device memory pointer for rectangle objects
rectangleObject* d_rects;

//TODO: if this method has to accept any type of objects, then the std vector 
// must store some kind of type erased objects
void uploadTodevice(rectangleObject* h_rects, const size_t numObj){
    util::my_uploadToDevice(&d_rects, h_rects, numObj);   
}

void testdownloadToHost(rectangleObject* h_rects, const size_t num_rects){
    util::my_testDownloadToHost(h_rects, d_rects, num_rects);
}

void computeTree(size_t num_rects, int blocksPerGrid, int threadsPerBlock){
    computeAABBs<<<blocksPerGrid, threadsPerBlock>>>(num_rects);
}
__global__ void computeAABBs(const size_t num_rects){
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
}
