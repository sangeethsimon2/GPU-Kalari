#include "AABB.h"
#include "bvh.cuh"
#include "mycudaUtilities.cuh"
#include <cuda_runtime.h>

void version(){
std::cout << "Thrust version: " << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << "\n";
}
//Device memory pointer for rectangle objects
rectangleObject* d_rects;

//Device memory pointer for AABB objects
AABB* d_aabbs;

//Device memory pointer for Centroid objects 
CentroidX* d_centroidX;

//Device memory pointer for Centroid objects 
CentroidY* d_centroidY;

//Device memory pointer for Centroid objects 
CentroidZ* d_centroidZ;


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
__global__ void computeAABBs(T* d_objs, CentroidX* d_centroidX, CentroidY* d_centroidY, CentroidZ* d_centroidZ,
                             const size_t numObjs, AABB* d_aabbs){
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
(originX, originY, originZ)
*/
        m_aabbOfEachObject.minX = m_eachObj.originX;
        m_aabbOfEachObject.minY = m_eachObj.originY;
        m_aabbOfEachObject.minZ = m_eachObj.originZ;

        m_aabbOfEachObject.maxX = m_eachObj.originX + m_eachObj.width;
        m_aabbOfEachObject.maxY = m_eachObj.originY + m_eachObj.height;
        m_aabbOfEachObject.maxZ = m_eachObj.originZ + m_eachObj.depth; 

        d_centroidX->centroidX = (m_aabbOfEachObject.minX + m_aabbOfEachObject.maxX)* 0.5f;
        d_centroidY->centroidY = (m_aabbOfEachObject.minY + m_aabbOfEachObject.maxY)* 0.5f;
        d_centroidZ->centroidZ = (m_aabbOfEachObject.minZ + m_aabbOfEachObject.maxZ)* 0.5f;

        d_aabbs[idx] = m_aabbOfEachObject;
    }
}

void computeTree(size_t numObjs, int blocksPerGrid, int threadsPerBlock){
    

    util::allocateMemoryOndevice(d_aabbs, numObjs, sizeof(AABB));
    util::allocateMemoryOndevice(d_centroidX, numObjs, sizeof(CentroidX));
    util::allocateMemoryOndevice(d_centroidY, numObjs, sizeof(CentroidY));
    util::allocateMemoryOndevice(d_centroidZ, numObjs, sizeof(CentroidZ));

    
    computeAABBs<<<blocksPerGrid, threadsPerBlock>>>(d_rects, d_centroidX, d_centroidY, d_centroidZ, 
        numObjs, d_aabbs);
    


    auto td_centroidX = thrust::device_pointer_cast<CentroidX>(d_centroidX);
    auto td_centroidY = thrust::device_pointer_cast<CentroidY>(d_centroidY);
    auto td_centroidZ = thrust::device_pointer_cast<CentroidZ>(d_centroidZ);

    auto minmaxX = thrust::minmax_element(td_centroidX, td_centroidX+numObjs);
    auto minmaxY = thrust::minmax_element(td_centroidY, td_centroidY+numObjs);
    auto minmaxZ = thrust::minmax_element(td_centroidZ, td_centroidZ+numObjs);

}


//Expliciit instantiations
template void uploadTodevice<rectangleObject>(rectangleObject*, const size_t);
template void testdownloadToHost<rectangleObject>(rectangleObject*, const size_t);
