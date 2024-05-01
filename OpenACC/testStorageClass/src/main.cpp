#include <iostream>
#include <memory>
#include <vector>

#include "state.hpp"

int main(){
   State state(100);
   state.copyClassToDevice();
   state.initializeState();

   //Print the modified vector
   for (int i = 0; i < state.getTotalSize(); ++i)  {
        std::cout << state.getStateAtPoint(i) << " ";
    }
    std::cout << std::endl;

   #pragma acc parallel loop independent
              for (int i = 0; i < state.getTotalSize(); ++i) {
                state.getStateAtPoint(i) = 4;
              }

   //Print the modified vector
   for (int i = 0; i < state.getTotalSize(); ++i)  {
        std::cout << state.getStateAtPoint(i) << " ";
    }
    std::cout << std::endl;
    return 0;
}