#pragma once 

#include "AABB.h"
#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

template<typename T>
__global__ void computeAABBs( T*,  const size_t );

struct leftBottomBoundCoordinates{
     float x{0.}; float y{0.}; float z{0.};
};

struct rightTopBoundCoordinates{
     float x{0.}; float y{0.}; float z{0.};
};

struct CentroidX{
     float value=-1000.;
     uint32_t mortonCode = 0;

     __host__ __device__
     bool operator < (const CentroidX& other)const {return this->value< other.value;}
};

struct CentroidY{
     float  value=-1000.;
     uint32_t mortonCode = 0;
     __host__ __device__
     bool operator < (const CentroidY& other)const {return this->value< other.value;}
};


struct CentroidZ{
     float value=-1000.;
     uint32_t mortonCode = 0;
    
     __host__ __device__
     bool operator < (const CentroidZ& other)const {return this->value< other.value;}
};
