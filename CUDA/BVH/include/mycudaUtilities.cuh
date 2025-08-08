#include <iostream>
#include "AABB.h"


namespace util{
    //Error related APIS
    #define checkCudaErrors(val) check_cuda_error((val), #val, __FILE__, __LINE__)
    void check_cuda_error(cudaError_t result, const char *const func, const char *const file, int const line){
       if(result!=cudaSuccess){
        std::cerr<<"CUDA error at "<<file<<":"<<line<<" with code="<<result<<" (" << cudaGetErrorString(result) << ") \"" << func << "\"" << std::endl;
        exit(EXIT_FAILURE);
        }    
    }
    //Memory related APIS    
    template<typename T>
    void my_cudamallocT(T** _deviceResource, const size_t _resourceCount, const size_t _resourceSize){
        checkCudaErrors(cudaMalloc((void**)_deviceResource, _resourceCount * _resourceSize));
    }

    template<typename T>
    void my_cudaMemcpyT(T* _destination, T* _source, const size_t _resourceSize, cudaMemcpyKind _copyType){
        checkCudaErrors(cudaMemcpy(_destination, _source, _resourceSize, _copyType));
    }

    template<typename T>
    void my_cudafreeT(T* _deviceResource){
        checkCudaErrors(cudaFree(_deviceResource));
    }

    // Connecter functions between C++ & CUDA APIS
    void my_uploadToDevice(rectangleObject** d_rects, rectangleObject* h_rects, const size_t numObj){
        my_cudamallocT(d_rects, numObj, sizeof(rectangleObject)); 
        std::cout<<"Allocated memory\n";
        my_cudaMemcpyT( *d_rects, h_rects, numObj * sizeof(rectangleObject), cudaMemcpyHostToDevice);
        std::cout<<"Finished uploading memory from host to device\n";
    }
   
    void my_testDownloadToHost(rectangleObject* h_rects, rectangleObject* d_rects, const size_t numObjs){ 
        if(h_rects != NULL && d_rects != NULL){
            my_cudaMemcpyT( h_rects, d_rects, numObjs * sizeof(rectangleObject), cudaMemcpyDeviceToHost);
            std::cout<<"Finished downloading memory from device to host\n";
        }
        my_cudafreeT(d_rects);
        std::cout<<"Deallocated memory\n";
    }
    
}
