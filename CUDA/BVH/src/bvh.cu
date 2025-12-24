#pragma once 

#include "bvh.cuh"
#include "mycudaUtilities.cuh"

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

//Device memory pointer for morton code array for all primitives 
uint32_t* d_mortonCodeArrayForPrimitives;


template<typename T>
void uploadTodevice(T* h_rects, const size_t numObj){
    util::my_uploadToDevice(&d_rects, h_rects, numObj);  // This could be error prone when T! typeof d_rects 
}

template<typename T>
void testdownloadToHost(T* h_rects, const size_t numObjs){
    util::my_testDownloadToHost(h_rects, d_rects, numObjs);
}

/*
template<typename T>
void downloadToHost(T* h_objs, const size_t numObjs){
    util::my_downloadToHost(h_objs, numObjs);
}*/

/*
template<typename T>
void downloadToHost(T* h_objs, const size_t numObjs){
    util::my_downloadToHost(h_objs, numObjs);
}*/

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

        d_minCoord[idx].y = m_eachObj.originY;
        d_minCoord[idx].x = m_eachObj.originX;
        d_minCoord[idx].z = m_eachObj.originZ;

        d_maxCoord[idx].x = m_eachObj.originX + m_eachObj.width;
        d_maxCoord[idx].y = m_eachObj.originY + m_eachObj.height;
        d_maxCoord[idx].z = m_eachObj.originZ + m_eachObj.depth; 

        d_centroidX[idx].value = (d_minCoord[idx].x + d_maxCoord[idx].x)* 0.5f;
        d_centroidY[idx].value = (d_minCoord[idx].y + d_maxCoord[idx].y)* 0.5f;
        d_centroidZ[idx].value = (d_minCoord[idx].z + d_maxCoord[idx].z)* 0.5f;

    }
}

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ unsigned int expandBits(unsigned int _input){
    // TODO: Currently hardcoded for 32 bit integer
    _input = (_input * 0x00010001u) & 0xFF0000FFu; //16 bit shift
    _input = (_input * 0x00000101u) & 0x0F00F00Fu; //8 bit shift
    _input = (_input * 0x00000011u) & 0xC30C30C3u; //4 bit shift
    _input = (_input * 0x00000005u) & 0x49249249u; //2 bit shift
    return _input;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int computeMortonCode3D(float _centroidX, float _centroidY, float _centroidZ){

    //Convert floating point to fixed point integers
    unsigned int fixedPointCentroidX = min(max(_centroidX * 1024.0f, 0.0f), 1023.0f);
    unsigned int fixedPointCentroidY = min(max(_centroidY * 1024.0f, 0.0f), 1023.0f);
    unsigned int fixedPointCentroidZ = min(max(_centroidZ * 1024.0f, 0.0f), 1023.0f);

    //Expand the bits to create two extra unfilled bits per bit to facilitate interleaving     
    unsigned int fixedPointCentroidXExpanded = expandBits(fixedPointCentroidX);
    unsigned int fixedPointCentroidYExpanded = expandBits(fixedPointCentroidY);
    unsigned int fixedPointCentroidZExpanded = expandBits(fixedPointCentroidZ);

    return fixedPointCentroidXExpanded*4 + fixedPointCentroidYExpanded*2 + fixedPointCentroidZExpanded;
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
    size_t numObjs, uint32_t* d_mortonCodeArrayForPrimitives){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numObjs){
        //Normalize the centroid values
        d_centroidX[idx].value = (d_centroidX[idx].value-minX)/(maxX-minX);
        d_centroidY[idx].value = (d_centroidY[idx].value-minY)/(maxY-minY);
        d_centroidZ[idx].value = (d_centroidZ[idx].value-minZ)/(maxZ-minZ);

        //Compute mortoncode for this coordinate
        unsigned int mortoncodeTemp = computeMortonCode3D(d_centroidX[idx].value, d_centroidY[idx].value, d_centroidZ[idx].value);
        d_mortonCodeArrayForPrimitives[idx] = mortoncodeTemp; 
    }
}

void computeTree( SmartMemoryManager::HostDeviceMemoryManager<rectangleObject> rect_smartBuffer,
    const int blocksPerGrid, const int threadsPerBlock){
    
    const size_t numObjs = rect_smartBuffer.getSize();

    /* Memory allocations*/
    util::allocateMemoryOndevice(d_minCoord, numObjs, sizeof(leftBottomBoundCoordinates));
    util::allocateMemoryOndevice(d_maxCoord, numObjs, sizeof(rightTopBoundCoordinates));
    util::allocateMemoryOndevice(d_centroidX, numObjs, sizeof(CentroidX));
    util::allocateMemoryOndevice(d_centroidY, numObjs, sizeof(CentroidY));
    util::allocateMemoryOndevice(d_centroidZ, numObjs, sizeof(CentroidZ));
   
    util::allocateMemoryOndevice(d_mortonCodeArrayForPrimitives, numObjs, sizeof(uint32_t));

    computeMinMaxBoundsAndCentroids<<<blocksPerGrid, threadsPerBlock>>>(rect_smartBuffer.getDevicePointer(), d_minCoord, d_maxCoord, 
        d_centroidX, d_centroidY, d_centroidZ, numObjs);

    //TO be investigated whether this is done natively on GPU following the computeAABB kernel 
    //or is there a host-gpu transfer
    auto td_centroidX = thrust::device_pointer_cast<CentroidX>(d_centroidX);
    auto td_centroidY = thrust::device_pointer_cast<CentroidY>(d_centroidY);
    auto td_centroidZ = thrust::device_pointer_cast<CentroidZ>(d_centroidZ);

    auto minmaxX = thrust::minmax_element(td_centroidX, td_centroidX+numObjs);
    auto minmaxY = thrust::minmax_element(td_centroidY, td_centroidY+numObjs);
    auto minmaxZ = thrust::minmax_element(td_centroidZ, td_centroidZ+numObjs);

    //Debugging:
    CentroidX minX_value; 
    thrust::copy(minmaxX.first, minmaxX.first+1, &minX_value);
    CentroidX maxX_value; 
    thrust::copy(minmaxX.second, minmaxX.second+1, &maxX_value);
    CentroidY minY_value; 
    thrust::copy(minmaxY.first, minmaxY.first+1, &minY_value);
    CentroidY maxY_value; 
    thrust::copy(minmaxY.second, minmaxY.second+1, &maxY_value);
    CentroidZ minZ_value; 
    thrust::copy(minmaxZ.first, minmaxZ.first+1, &minZ_value);
    CentroidZ maxZ_value; 
    thrust::copy(minmaxZ.second, minmaxZ.second+1, &maxZ_value);
    
    computeScaledCentroids<<<blocksPerGrid, threadsPerBlock>>>(d_centroidX, 
    d_centroidY, d_centroidZ, 
    minX_value.value, maxX_value.value,
    minY_value.value, maxY_value.value,
    minZ_value.value, maxZ_value.value,
    numObjs, d_mortonCodeArrayForPrimitives 
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
