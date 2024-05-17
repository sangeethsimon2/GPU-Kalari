/*Class that implements the dirichlet boundary condition*/
/*This class implements the specific 'update' method
that will be called by the factory method in the BC updater class*/

#ifndef _DIRICHLETIMPL_H
#define _DIRICHLETIMPL_H

#include<iostream>
#include <cstdio>
#include "boundaryConditionTypeInterface.hpp"

template<int DIM>
class DirichletImpl: public BoundaryConditionTypeInterface<DirichletImpl<DIM>>{

    public:
            //CTOR
            DirichletImpl(){
                printf("DirichletImpl CTOR\n");
                copyClassToDevice();
            }
            ~DirichletImpl(){
                removeClassFromDevice();
            }
            void updateBoundaries(const int _Nx){
                printf("Call updateBoundaries in DirichletImpl\n");

            }
            void copyClassToDevice(){
              #pragma acc enter data copyin(this[0:1])
            }
            void removeClassFromDevice(){
             #pragma acc exit data delete(this[0:1])
            }

};

template class DirichletImpl<2>;

#endif