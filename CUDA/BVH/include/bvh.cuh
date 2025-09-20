#pragma once 

#include "AABB.h"
#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

template<typename T>
__global__ void computeAABBs( T*,  const size_t );


struct CentroidX{
     float centroidX=-1.;

     __host__ __device__
     bool operator < (const CentroidX& other)const {return this->centroidX< other.centroidX;}
};

struct CentroidY{
     float centroidY=-1.;
    
     __host__ __device__
     bool operator < (const CentroidY& other)const {return this->centroidY< other.centroidY;}
};


struct CentroidZ{
     float centroidZ=-1.;
    
     __host__ __device__
     bool operator < (const CentroidZ& other)const {return this->centroidZ< other.centroidZ;}
};
