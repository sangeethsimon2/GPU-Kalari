#pragma once 

#include <iostream>
#include "AABB.h"
#include <cuda_runtime.h>
//#include "bvh.cuh"


namespace util{
    //Error related APIS
    #define checkCudaErrors(val) check_cuda_error((val), #val, __FILE__, __LINE__)
    inline void check_cuda_error(cudaError_t result, const char *const func, const char *const file, int const line){
       if(result!=cudaSuccess){
        std::cerr<<"CUDA error at "<<file<<":"<<line<<" with code="<<result<<" (" << cudaGetErrorString(result) << ") \"" << func << "\"" << std::endl;
        exit(EXIT_FAILURE);
        }    
    }
    //Memory related APIS    
    template<typename T>
    void my_cudamallocT(T** _deviceResource, const size_t _resourceCount, const size_t _unitResourceSize){
        checkCudaErrors(cudaMalloc((void**)_deviceResource, _resourceCount * _unitResourceSize));
    }

    template<typename T>
    void my_cudaMemcpyT(T* _destination, T* _source, const size_t _totalResourceSize, cudaMemcpyKind _copyType){
        checkCudaErrors(cudaMemcpy(_destination, _source, _totalResourceSize, _copyType));
    }

    template<typename T>
    void my_cudafreeT(T* _deviceResource){
        checkCudaErrors(cudaFree(_deviceResource));
        _deviceResource = nullptr;
    }

    // Connecter functions between C++ & CUDA APIS
    
    //Create memory only 
    template<typename T>
    void allocateMemoryOndevice(T*& _deviceResource, const size_t _resourceCount, const size_t _unitResourceSize){
        my_cudamallocT(&_deviceResource, _resourceCount, sizeof(T));
        std::cout<<"Allocated memory\n";
    } 

    //Deallocate memory only 
    template<typename T>
    void deallocateMemoryFromdevice(T*& _deviceResource){
        my_cudafreeT(_deviceResource);
        std::cout<<"Deallocated memory\n";
    }

    //Create memory and copy to device 
    template<typename T>
    void my_uploadToDevice(T** d_rects, T* h_rects, const size_t numObj){
        my_cudamallocT(d_rects, numObj, sizeof(T)); 
        std::cout<<"Allocated memory\n";
        my_cudaMemcpyT( *d_rects, h_rects, numObj * sizeof(T), cudaMemcpyHostToDevice);
        std::cout<<"Finished uploading memory from host to device\n";
    }
   
    //test download
    inline void my_testDownloadToHost(rectangleObject* h_rects, rectangleObject* d_rects, const size_t numObjs){ 
        if(h_rects != NULL && d_rects != NULL){
            my_cudaMemcpyT( h_rects, d_rects, numObjs * sizeof(rectangleObject), cudaMemcpyDeviceToHost);
            std::cout<<"Finished downloading memory from device to host\n";
        }
        my_cudafreeT(d_rects);
        std::cout<<"Deallocated memory\n";
    }
    
   template <typename T> 
   void my_downloadToHost(T* h_objs, T* d_objs, const size_t numObjs){
        if(h_objs != nullptr && d_objs != nullptr){
            my_cudaMemcpyT(h_objs, d_objs, numObjs * sizeof(T), cudaMemcpyDeviceToHost);
            std::cout<<"Finished downloading memory from device to host\n";
        }
        my_cudafreeT(d_objs);
        std::cout<<"Deallocated memory\n";
   } 
}
