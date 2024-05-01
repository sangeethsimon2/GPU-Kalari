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
         Kernel(std::shared_ptr<Strategy> &&_strategy): m_strategy(std::move(_strategy)){}

        void updateSolution(std::vector<int>& _solution){
            m_strategy->updateSolution(_solution);
        }
    protected:
         std::shared_ptr<Strategy>m_strategy;
};
#endif