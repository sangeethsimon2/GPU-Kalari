#include "gpuTimer.hpp"


void gpuTimer::startClock(){
    cudaEventRecord(startRec,0);
}
void gpuTimer::stopClock(){
    cudaEventRecord(stopRec,0);
    cudaEventSynchronize(stopRec);
    float timeElapsed;
    cudaEventElapsedTime(&timeElapsed, startRec, stopRec);
    m_totalElapsedTime +=(double)1e-3 *timeElapsed;
}

double gpuTimer::elapsedTime() const{
return m_totalElapsedTime;
}

void gpuTimer::resetClock(){
    m_totalElapsedTime = 0.;
}