#include "AABB.h"
#include "bvh.cuh"
#include "mycudaUtilities.cuh"
#include <cuda_runtime.h>

//Device memory pointer for rectangle objects
rectangleObject* d_rects;

//Device memory pointer for AABB objects
AABB* d_aabbs;

template<typename T>
void uploadTodevice(T* h_rects, const size_t numObj){
    util::my_uploadToDevice(&d_rects, h_rects, numObj);  // This could be error prone when T! typeof d_rects 
}

void testdownloadToHost(rectangleObject* h_rects, const size_t numObjs){
    util::my_testDownloadToHost(h_rects, d_rects, numObjs);
}

template<typename T>
__global__ void computeAABBs(T* d_rects, const size_t numObjs, AABB* d_aabbs){
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if(idx < numObjs){
        d_rects[idx].originX=0.;
     }
}
void computeTree(size_t numObjs, int blocksPerGrid, int threadsPerBlock){
    util::allocateMemoryOndevice(d_aabbs, numObjs, sizeof(AABB));
    computeAABBs<<<blocksPerGrid, threadsPerBlock>>>(d_rects, numObjs, d_aabbs);
}


