/* This class represents the 'context' class in the strategy design pattern that is used to dispatch
the correct jacobi implementation to compute the heat equation updates*/

#ifndef _KERNEL_H
#define _KERNEL_H

#include<memory>

#include "strategyA.hpp"
#include "strategyB.hpp"

//Declare and define a template class with a single template parameter corresponding to a strategy type
template<typename Strategy>
class Kernel{
    public:
         Kernel(std::shared_ptr<Strategy> &&_strategy): m_strategy(std::move(_strategy)){
            copyClassToDevice();
         }

        void updateSolution(std::vector<int>& _solution){
            m_strategy->updateSolution(_solution);
         }

         //Copy to device method that copies :this
         void copyClassToDevice(){
            #pragma acc enter data copyin(this[0:1], m_strategy[0:1])
          }
         void removeClassFromDevice(){
            #pragma acc exit data delete(this[0:1], m_strategy[0:1])
         }
         ~Kernel(){
            removeClassFromDevice();
         }
    protected:
         std::shared_ptr<Strategy>m_strategy;
};
#endif