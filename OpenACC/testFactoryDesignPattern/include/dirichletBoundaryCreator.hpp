/*Class that creates the object that implements the dirichlet boundary condition.
Part of the factor method patter covering the bondary condition implementation*/

#ifndef _DIRICHLETBOUNDARYCREATOR_H
#define _DIRICHLETBOUNDARYCREATOR_H

#include<iostream>

#include "boundaryCreatorInterface.hpp"
#include "dirichletBoundaryTypeImpl.hpp"
#include "boundaryConditionTypeInterface.hpp"
template<typename DirichletImpl>
class DirichletBoundaryCreator: public BoundaryCreatorInterface<DirichletBoundaryCreator<DirichletImpl>, DirichletImpl>{
    public:

            using BaseClass = BoundaryCreatorInterface<DirichletBoundaryCreator<DirichletImpl>, DirichletImpl>;

            //CTOR
            DirichletBoundaryCreator(int _Nx): BaseClass(_Nx)
            {printf("DirichletBoundaryCreator CTOR\n");
             copyClassToDevice();
            }
            ~DirichletBoundaryCreator(){
                removeClassFromDevice();
            }
            std::shared_ptr<BoundaryConditionTypeInterface<DirichletImpl>> createBoundaryType(){
                std::cout<<" DirichletBoundaryCreator createBoundaryType called\n";
                return( std::make_shared<BoundaryConditionTypeInterface<DirichletImpl>>());
            };
            void copyClassToDevice(){
            #pragma acc enter data copyin(this[0:1])
           }
           void removeClassFromDevice(){
            #pragma acc exit data delete(this[0:1])
           }

};

#endif