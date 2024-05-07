/*Class that implements the jacobi method interface with a serial algorithm*/
/*This class represents the specific 'strategy' and
participates in the strategy design pattern by interacting with its
base class, the kernel class and the main()*/

#ifndef _STRATEGYB_H
#define _STRATEGYB_H

#include<iostream>
#include <math.h>
#include <fstream>
#include <iomanip>

//Define a template class with a single non-type template parameter corresponding to dimension
template<int DIM>
class StrategyB{

    public:
           //CTOR
           StrategyB(){
              copyClassToDevice();
           }
            ~StrategyB(){
            removeClassFromDevice();
           }

           //#pragma acc routine
           void updateSolution(std::vector<int>& _solution){
              if (acc_on_device(acc_device_host))
                  printf(" The kernel is on host %d \n", acc_on_device(acc_device_host));
              else if (acc_on_device(acc_device_nvidia))
                  printf(" The kernel is on device %d \n", acc_on_device(acc_device_nvidia));
              printf(" Updating solution using Strategy B for dim=%d \n", DIM);

              #pragma acc loop
              for(int i = 0; i < _solution.size(); i++){
                _solution[i] +=3;
              }
           }
           void copyClassToDevice(){
            #pragma acc enter data copyin(this[0:1])
          }
          void removeClassFromDevice(){
            #pragma acc exit data delete(this[0:1])
         }
};
#endif