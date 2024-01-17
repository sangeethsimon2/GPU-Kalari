#include "hostTimer.hpp"

void hostTimer::startClock(){
    m_startTime = timerType_t::now();
}
void hostTimer::stopClock(){
    m_totalElapsedTime += std::chrono::duration_cast<std::chrono::nanoseconds> (timerType_t::now()-m_startTime);
}
double hostTimer::elapsedTime() const{
    return (m_totalElapsedTime.count()*1e-9);
}
void hostTimer::resetClock(){
   m_totalElapsedTime = 0s;
}