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

//Device memory pointer for min coordinate of each object 
leftBottomBoundCoordinates* d_minCoord;

//Device memory pointer for max coordinate of each object 
rightTopBoundCoordinates* d_maxCoord; 



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

/*
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
/*
        m_aabbOfEachObject.minX = m_eachObj.originX;
        m_aabbOfEachObject.minY = m_eachObj.originY;
        m_aabbOfEachObject.minZ = m_eachObj.originZ;

        m_aabbOfEachObject.maxX = m_eachObj.originX + m_eachObj.width;
        m_aabbOfEachObject.maxY = m_eachObj.originY + m_eachObj.height;
        m_aabbOfEachObject.maxZ = m_eachObj.originZ + m_eachObj.depth; 

        m_aabbOfEachObject.centroidX = (m_aabbOfEachObject.minX + m_aabbOfEachObject.maxX)* 0.5f;
        m_aabbOfEachObject.centroidY = (m_aabbOfEachObject.minY + m_aabbOfEachObject.maxY)* 0.5f;
        m_aabbOfEachObject.centroidZ = (m_aabbOfEachObject.minZ + m_aabbOfEachObject.maxZ)* 0.5f;


        d_centroidX->centroidX = (m_aabbOfEachObject.minX + m_aabbOfEachObject.maxX)* 0.5f;
        d_centroidY->centroidY = (m_aabbOfEachObject.minY + m_aabbOfEachObject.maxY)* 0.5f;
        d_centroidZ->centroidZ = (m_aabbOfEachObject.minZ + m_aabbOfEachObject.maxZ)* 0.5f;

        d_aabbs[idx] = m_aabbOfEachObject;
    }
}
*/

template<typename T>
__global__ void computeMinMaxBoundsAndCentroids(T* d_objs, leftBottomBoundCoordinates* d_minCoord, 
    rightTopBoundCoordinates* d_maxCoord,
    CentroidX* d_centroidX, CentroidY* d_centroidY, CentroidZ* d_centroidZ,
    const size_t numObjs){
 /*
        <--  width  -->
        --------------- ^
        |             | |
        |             | | height
        |             | |            
        X-------------- v 
(originX, originY, originZ)
*/
    //Can we launch a 2D or 3D grid of threads to do this?
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numObjs){
        T m_eachObj = d_objs[idx];

        d_minCoord[idx].x = m_eachObj.originX;
        d_minCoord[idx].y = m_eachObj.originY;
        d_minCoord[idx].z = m_eachObj.originZ;

        d_maxCoord[idx].x = m_eachObj.originX + m_eachObj.width;
        d_maxCoord[idx].y = m_eachObj.originY + m_eachObj.height;
        d_maxCoord[idx].z = m_eachObj.originZ + m_eachObj.depth; 

        d_centroidX[idx].value = (d_minCoord[idx].x + d_maxCoord[idx].x)* 0.5f;
        d_centroidY[idx].value = (d_minCoord[idx].y + d_maxCoord[idx].y)* 0.5f;
        d_centroidZ[idx].value = (d_minCoord[idx].z + d_maxCoord[idx].z)* 0.5f;

    }
}
/**
 * We could have instead of float values for minX, maxX etc, passed CentroidX* etc and with an operload of 
 * operators + and - achieved the normalization
 */
__global__ void computeScaledCentroids(CentroidX* d_centroidX, 
    CentroidY* d_centroidY, CentroidZ* d_centroidZ,
     float minX, float maxX,
    float minY, float maxY, 
    float minZ, float maxZ,
    size_t numObjs){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numObjs){
        //Normalize the centroid values
        d_centroidX[idx].value = (d_centroidX[idx].value-minX)/(maxX-minX);
        d_centroidY[idx].value = (d_centroidY[idx].value-minY)/(maxY-minY);
        d_centroidZ[idx].value = (d_centroidZ[idx].value-minZ)/(maxZ-minZ);
    /*
        //Convert floating point to fixed point integers
        uint32_t fixedPointCentroidX = min(max(d_centroidX[idx].value * 1024.0f, 0.0f), 1023.0f);
        uint32_t fixedPointCentroidX = min(max(d_centroidY[idx].value * 1024.0f, 0.0f), 1023.0f);
        uint32_t fixedPointCentroidX = min(max(d_centroidZ[idx].value * 1024.0f, 0.0f), 1023.0f);
       
        //Expand bits 
        //I want the expand bit code to be non-blocking be called by all thread simultaneously at this point
    */
    }
}

void computeTree(size_t numObjs, int blocksPerGrid, int threadsPerBlock){
    
    /* Memory allocations*/
    util::allocateMemoryOndevice(d_minCoord, numObjs, sizeof(leftBottomBoundCoordinates));
    util::allocateMemoryOndevice(d_maxCoord, numObjs, sizeof(rightTopBoundCoordinates));
    util::allocateMemoryOndevice(d_centroidX, numObjs, sizeof(CentroidX));
    util::allocateMemoryOndevice(d_centroidY, numObjs, sizeof(CentroidY));
    util::allocateMemoryOndevice(d_centroidZ, numObjs, sizeof(CentroidZ));
    
    computeMinMaxBoundsAndCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_rects, d_minCoord, d_maxCoord, 
        d_centroidX, d_centroidY, d_centroidZ, numObjs);

    //TO be investigated whether this is done natively on GPU following the computeAABB kernel 
    //or is there a host-gpu transfer
    auto td_centroidX = thrust::device_pointer_cast<CentroidX>(d_centroidX);
    auto td_centroidY = thrust::device_pointer_cast<CentroidY>(d_centroidY);
    auto td_centroidZ = thrust::device_pointer_cast<CentroidZ>(d_centroidZ);

    auto minmaxX = thrust::minmax_element(td_centroidX, td_centroidX+numObjs);
    auto minmaxY = thrust::minmax_element(td_centroidY, td_centroidY+numObjs);
    auto minmaxZ = thrust::minmax_element(td_centroidZ, td_centroidZ+numObjs);

    computeScaledCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_centroidX, 
    d_centroidY, d_centroidZ, 
    minmaxX.first.get()->value, minmaxX.second.get()->value,
    minmaxY.first.get()->value, minmaxY.second.get()->value,
    minmaxZ.first.get()->value, minmaxZ.second.get()->value,
    numObjs
    );
/*    
    util::allocateMemoryOndevice(d_aabbs, numObjs, sizeof(AABB));
    computeAABBs<<<blocksPerGrid, threadsPerBlock>>>(d_rects, d_centroidX, d_centroidY, d_centroidZ, 
        numObjs, d_aabbs);
  */  


}


//Expliciit instantiations
template void uploadTodevice<rectangleObject>(rectangleObject*, const size_t);
template void testdownloadToHost<rectangleObject>(rectangleObject*, const size_t);
