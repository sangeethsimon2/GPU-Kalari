#ifndef MATOPERONGPUCUDA
#define MATOPERONGPUCUDA

#include <iostream>
#include <stdio.h>
#include <cassert>


#include <cuda.h>
#include <cuda_runtime.h>

#define tileWidth 16

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      std::fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void printCudaVersionNumber();
void uploadToDevice(const size_t size, const double* src, double** dst);
void downloadToHost(const size_t size, const double* src, double* dst);
void matrixMultiplyOnDevice( double**, double**, double**, const int);
void matrixMultiplyOnDeviceUsingSharedMem( double**, double**, double**, const int);
void freeDeviceMemory(double**, double**, double**);
#endif
