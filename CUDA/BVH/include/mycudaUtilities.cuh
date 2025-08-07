#include <iostream>
#include "AABB.h"


//TODO: remove dependency on rectangleObject in all APIs
namespace util{
    #define checkCudaErrors(val) check_cuda_error((val), #val, __FILE__, __LINE__)
    void check_cuda_error(cudaError_t result, const char *const func, const char *const file, int const line){
       if(result!=cudaSuccess){
        std::cerr<<"CUDA error at "<<file<<":"<<line<<" with code="<<result<<" (" << cudaGetErrorString(result) << ") \"" << func << "\"" << std::endl;
        exit(EXIT_FAILURE);
        }    
    }
    
    void my_cudamalloc(rectangleObject** d_rects, const size_t numObj){
        checkCudaErrors(cudaMalloc((void**)d_rects, numObj * sizeof(rectangleObject)));
    }

    void my_cudaMemcpy(rectangleObject* dest, rectangleObject* src, size_t size, cudaMemcpyKind copyType){
       checkCudaErrors(cudaMemcpy(dest, src, size, copyType)); 
    }

    void my_uploadToDevice(rectangleObject** d_rects, rectangleObject* h_rects, const size_t numObj){
        my_cudamalloc(d_rects, numObj); 
        std::cout<<"Allocated memory\n";
        my_cudaMemcpy( *d_rects, h_rects, numObj * sizeof(rectangleObject), cudaMemcpyHostToDevice);
        std::cout<<"Finished uploading memory from host to device\n";
    }


    /*
    void uploadToDevice(const size_t size, const double* src, double** dst){
    gpuErrCheck(cudaMalloc((void**)dst, size));
    std::cout<<"Allocated memory\n";
    gpuErrCheck(cudaMemcpy(*dst, src, size, cudaMemcpyHostToDevice));
    std::cout<<"Finished uploading memory from host to device\n";
    };
    */
    
    //TODO: remove dependency on rectangleObject
    void my_cudafree(rectangleObject *src){
        checkCudaErrors( cudaFree(src));
    }

    void my_testDownloadToHost(rectangleObject* h_rects, rectangleObject* d_rects, const size_t numObjs){ 
        if(h_rects != NULL && d_rects != NULL){
            my_cudaMemcpy( h_rects, d_rects, numObjs * sizeof(rectangleObject), cudaMemcpyDeviceToHost);
            std::cout<<"Finished downloading memory from device to host\n";
        }
        my_cudafree(d_rects);
        std::cout<<"Deallocated memory\n";
    }


    
}
