#include "bvh.h"
#include "bvh.cuh"

void computeTree(int num_rects, int blocksPerGrid, int threadsPerBlock){
    computeAABBs<<<blocksPerGrid, threadsPerBlock>>>(num_rects);
}
__global__ void computeAABBs(const int num_rects){
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
}
