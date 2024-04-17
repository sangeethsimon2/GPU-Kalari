/* A class that describes a timer class for cpu time measurements*/
/* Inspired from host time class written by pkestner https://github.com/pkestene/cerfacs-training-kokkos/*/
#ifndef HOSTTIMER_H_
#define HOSTTIMER_H_

#include <chrono>
using namespace std::literals::chrono_literals;

class hostTimer{
  public:

  //Typedefs for various utilites from std::chrono
  //Type of clock
  using timerType_t = std::chrono::high_resolution_clock;
  //Type of time point for this clock
  using timePoint_t = timerType_t::time_point;
  //Type of time units
  using durationNanoSeconds_t  = std::chrono::nanoseconds;

  //Constructor
  hostTimer():m_startTime(0s), m_totalElapsedTime(0s) {}

  //Destructor
  ~hostTimer() = default;

  //Method to start the timer
  void startClock();
  void stopClock();
  double elapsedTime() const;
  void resetClock();

  protected:
  //Member to store the start time in number of ticks
  timePoint_t m_startTime;

  //Member to store the elapsed time in number of ticks
  durationNanoSeconds_t m_totalElapsedTime;
};
#endif