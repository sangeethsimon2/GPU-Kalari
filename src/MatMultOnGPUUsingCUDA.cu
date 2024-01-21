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

void freeDeviceMemory(double** Md, double** Nd, double** Pd){
    cudafree(*Md);
    cudafree(*Nd);
    cudafree(*Pd);
}