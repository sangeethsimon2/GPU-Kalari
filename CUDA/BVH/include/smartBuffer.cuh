#pragma once 

#include "AABB.h"
#include "mycudaUtilities.cuh"

namespace SmartMemoryManager{

    template<typename T>
    struct HostDeviceMemoryManager{
        public:
            HostDeviceMemoryManager(T* _ptr, size_t _count): h_ptr(_ptr), m_count(_count){};
            void uploadToDevice(){
                util::my_uploadToDevice(&d_ptr, h_ptr, m_count);
            }
            void downloadToHost(){
                util::my_downloadToHost(h_ptr, d_ptr, m_count);
            }
            T* getHostPointer(){return h_ptr;}
            const T* getHostPointer()const{return h_ptr;}
            T* getDevicePointer(){return d_ptr;}
            const T* getDevicePointer()const{return d_ptr;}
            size_t getSize()const{return m_count;}
            ~HostDeviceMemoryManager(){
                h_ptr=nullptr;
                m_count=0.;
                d_ptr=nullptr; 
            }
        private:
            T* h_ptr=nullptr;
            T* d_ptr=nullptr;
            size_t m_count{};
        
    };

}