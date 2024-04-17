/* A class that describes a timer class for gpu time measurements*/
/* Inspired from host time class written by pkestner https://github.com/pkestene/cerfacs-training-kokkos/*/
#ifndef GPUTIMER_H_
#define GPUTIMER_H_

#include <cuda_runtime.h>

class gpuTimer{
 public:

   gpuTimer(){
    cudaEventCreate(&startRec);
    cudaEventCreate(&stopRec);
    m_totalElapsedTime = 0.;
   }
   ~gpuTimer(){
    cudaEventDestroy(startRec);
    cudaEventDestroy(stopRec);
    }

   //Methods to start, stop, compute elapsed time and reset timer
   void startClock();
   void stopClock();
   double elapsedTime() const;
   void resetClock();

   protected:
   cudaEvent_t startRec, stopRec;
   double m_totalElapsedTime;

};

#endif
