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

template<typename T>
void testdownloadToHost(T* h_rects, const size_t numObjs){
    util::my_testDownloadToHost(h_rects, d_rects, numObjs);
}

template<typename T>
void downloadToHost(T* h_objs, const size_t numObjs){
    util::my_downloadToHost(h_objs, numObjs);
}

template<typename T>
__global__ void computeAABBs(T* d_objs, const size_t numObjs, AABB* d_aabbs){
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     AABB m_aabbOfEachObject; 
     if(idx < numObjs){
        T m_eachObj = d_objs[idx];
/*
        <--  width  -->
        --------------- ^
        |             | |
        |             | | height
        |             | |
        X-------------- v
(originX, originY)
*/
        m_aabbOfEachObject.minX = m_eachObj.originX;
        m_aabbOfEachObject.minY = m_eachObj.originY;
        m_aabbOfEachObject.minZ = m_eachObj.originZ;

        m_aabbOfEachObject.maxX = m_eachObj.originX + m_eachObj.width;
        m_aabbOfEachObject.maxY = m_eachObj.originY + m_eachObj.height;
        m_aabbOfEachObject.maxZ = m_eachObj.originZ + m_eachObj.depth; 
        
        d_aabbs[idx] = m_aabbOfEachObject;
    }
}

void computeTree(size_t numObjs, int blocksPerGrid, int threadsPerBlock){
    util::allocateMemoryOndevice(d_aabbs, numObjs, sizeof(AABB));
    computeAABBs<<<blocksPerGrid, threadsPerBlock>>>(d_rects, numObjs, d_aabbs);
}


//Expliciit instantiations
template void uploadTodevice<rectangleObject>(rectangleObject*, const size_t);
template void testdownloadToHost<rectangleObject>(rectangleObject*, const size_t);
