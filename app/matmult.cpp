#include<iostream>
#include<cstdlib>
#include<cassert>
#include "hostTimer.hpp"
#include "gpuTimer.hpp"
#include "MatrixBase.hpp"
#include "MatMultOnGPUUsingCUDA.cuh"



int main(int argc, char** argv){

    hostTimer hostTimer;
    gpuTimer gpuTimer;

    printCudaVersionNumber();


    if(std::atoi(argv[1])<=0 || std::atoi(argv[2])<=0){
        std::cerr<<" Invalid matrix sizes! They have to be >0!"<<std::endl;
        abort();
    }

    if(std::atoi(argv[1])!= std::atoi(argv[2])){
        std::cerr<<" We support only square matrices now!"<<std::endl;
        abort();
    }


    matrix M(std::atoi(argv[1]),std::atoi(argv[2]));
    //M.showMatrix();

    matrix N(std::atoi(argv[1]),std::atoi(argv[2]));
    //N.showMatrix();

    //solution matrix
    matrix P(std::atoi(argv[1]),std::atoi(argv[2]));
    P.initializeMatrixToZero();

    assert(M.getSizeInBytesOfMatrixElements()==N.getSizeInBytesOfMatrixElements());
    assert(P.getSizeInBytesOfMatrixElements()==M.getSizeInBytesOfMatrixElements());

    std::cout<<"Finished allocating the matrices\n";

    #ifdef ENABLE_SERIAL
    std::cout<<" Performing matrix mult on the cpu cores\n";

    hostTimer.startClock();
    //Call matrix multiplication
    P = M*N;
    //P.showMatrix();

    hostTimer.stopClock();
    std::cout<<"The elapsed time for CPU computation is: "<<hostTimer.elapsedTime()<<std::endl;
    #endif

    #ifdef ENABLE_CUDA
    //CUDA computations
    // Declare empty matrices that will be used on the device to recieve
    // matrices from CPU (Md, Nd) and send back the computed matrix (Pd) to CPU
    double *Md, *Nd, *Pd;

    uploadToDevice(M.getSizeInBytesOfMatrixElements(), M.getMatrixElements().data(), &Md);
    uploadToDevice(N.getSizeInBytesOfMatrixElements(), N.getMatrixElements().data(), &Nd);
    uploadToDevice(P.getSizeInBytesOfMatrixElements(), P.getMatrixElements().data(), &Pd);

    gpuTimer.startClock();

    matrixMultiplyOnDevice(&Md, &Nd, &Pd, M.getNumberOfElementsInMatrix());


    downloadToHost(P.getSizeInBytesOfMatrixElements(), Pd, P.getMatrixElements().data());

    gpuTimer.stopClock();
    std::cout<<"The elapsed time for GPU computation is: "<<gpuTimer.elapsedTime()<<std::endl;

    freeDeviceMemory(&Md, &Nd, &Pd);
    #endif

    return(0);

}