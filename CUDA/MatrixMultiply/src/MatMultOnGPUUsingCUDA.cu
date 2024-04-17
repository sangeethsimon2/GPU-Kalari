#include "MatMultOnGPUUsingCUDA.cuh"

void printCudaVersionNumber(){

    std::cout<<"CUDA compiled version:"<<__CUDACC_VER_MAJOR__<<std::endl;

    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout<<"CUDA runtime version:"<<runtime_ver<<std::endl;

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout<<"CUDA driver version:"<<driver_ver<<std::endl;

}
void uploadToDevice(const size_t size, const double* src, double** dst){
    gpuErrCheck(cudaMalloc((void**)dst, size));
    std::cout<<"Allocated memory\n";
    gpuErrCheck(cudaMemcpy(*dst, src, size, cudaMemcpyHostToDevice));
    std::cout<<"Finished uploading memory from host to device\n";
};

void downloadToHost(const size_t size,  const double* src, double*dst){
   gpuErrCheck(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    std::cout<<"Finished downloading memory from device to host\n";
}

void cudafree(double *src){
   gpuErrCheck( cudaFree(src));
}

__global__ void matrixMultiplyKernel( const double* Md, const double* Nd, double* Pd, const int width){
    const int tx =  blockIdx.x * tileWidth + threadIdx.x;
    const int ty = blockIdx.y * tileWidth + threadIdx.y;
    //printf("%d\n",tx);
    double pValue = 0.;
    for(auto k=0;k<width;k++){
        double MdElement = Md[ty*width+k];
        double NdElement = Nd[k*width+tx];
        pValue += MdElement*NdElement;
    }
    Pd[ty*width+tx] = pValue;
    //printf("%lf\n",pValue);
}

//CUDA kernel that uses shared memory to improve bandwidth
__global__ void matrixMultiplyKernel_usingSharedMem( const double* Md, const double* Nd, double* Pd, const int width){
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float Mds[tileWidth][tileWidth];
    __shared__ float Nds[tileWidth][tileWidth];

    //Row & Col of Pd element to compute
    const int row = by*tileWidth + ty;
    const int col = bx*tileWidth + tx;

   float pValue=0.;

   //Loop over the matrix in tileWidth blocks
   for (int gridStride=0; gridStride < width/tileWidth; ++gridStride){
     //Collaborative loading of elements from Md and Nd into Mds and Nds
     Mds[ty][tx] = Md[row*width + gridStride*tileWidth + tx];
     Nds[ty][tx] = Nd[(gridStride*tileWidth+ty)*width + col];

     //Thread sync to allow everry thread in the block to finish loading
     __syncthreads();

     for (int tileStride=0; tileStride < tileWidth; ++tileStride){
       pValue += Mds[ty][tileStride] * Nds[tileStride][tx];
     }

     //Thread sync to allow every thread to finish computing the partial sum
     // using the data in the shared mem
     __syncthreads();
   }
   //Store the element in the output matrix
   Pd[row*width+col] = pValue;

   }

void matrixMultiplyOnDevice( double** Md, double** Nd, double** Pd, const int widthOfMatrix){

    assert(*Md && *Nd && *Pd);
    dim3 dimGrid(widthOfMatrix/tileWidth, widthOfMatrix/tileWidth);
    dim3 dimBlock(tileWidth, tileWidth);
    std::cout<<"Launching the GPU kernel\n";
    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(*Md, *Nd, *Pd, widthOfMatrix);
    //matrixMultiplyKernel<<<2, 2>>>(*Md, *Nd, *Pd, widthOfMatrix);
    //gpuErrCheck(cudaDeviceSynchronize());
    gpuErrCheck(cudaGetLastError());
    std::cout<<"Finished eexcuting the GPU kernel\n";
}

void matrixMultiplyOnDeviceUsingSharedMem( double** Md, double** Nd, double** Pd, const int widthOfMatrix){

    assert(*Md && *Nd && *Pd);
    dim3 dimGrid(widthOfMatrix/tileWidth, widthOfMatrix/tileWidth);
    dim3 dimBlock(tileWidth, tileWidth);
    std::cout<<"Launching the GPU kernel\n";
    matrixMultiplyKernel_usingSharedMem<<<dimGrid, dimBlock, tileWidth*tileWidth*sizeof(double)>>>(*Md, *Nd, *Pd, widthOfMatrix);
    //matrixMultiplyKernel<<<2, 2>>>(*Md, *Nd, *Pd, widthOfMatrix);
    //gpuErrCheck(cudaDeviceSynchronize());
    gpuErrCheck(cudaGetLastError());
    std::cout<<"Finished eexcuting the GPU kernel\n";
}
void freeDeviceMemory(double** Md, double** Nd, double** Pd){
    cudafree(*Md);
    cudafree(*Nd);
    cudafree(*Pd);
}
