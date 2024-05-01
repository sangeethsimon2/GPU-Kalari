/*Class that implements the jacobi method interface with a serial algorithm*/
/*This class represents the specific 'strategy' and
participates in the strategy design pattern by interacting with its
base class, the kernel class and the main()*/

#ifndef _STRATEGYA_H
#define _STRATEGYA_H

#include<iostream>
#include <math.h>
#include <fstream>
#include <iomanip>
//Define a template class with a single non-type template parameter corresponding to dimension
template< int DIM>
class StrategyA{

    public:
           //CTOR
           StrategyA(){}

           void updateSolution(std::vector<int>& _solution){
            std::cout<<" Updating solution using Strategy A for dim "<<DIM<<"\n";
                for (int& eachelement : _solution)
                   eachelement+=2;
           }

};
#endif