#include "AABB.h"

template<typename T>
__global__ void computeAABBs( T*,  const size_t );