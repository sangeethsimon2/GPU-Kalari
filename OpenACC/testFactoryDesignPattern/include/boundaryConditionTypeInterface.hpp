/* Virtual base class for all variants of boundary condition implementations for the heat equation */
/*Part of the modified 'factory method design pattern' to call the correct updateboundary method
depending on the user choice using CRTP technique*/
#ifndef _BOUNDARYCONDITIONTYPEINTERFACE_H
#define _BOUNDARYCONDITIONTYPEINTERFACE_H

#include "dirichletBoundaryTypeImpl.hpp"
template <typename BoundaryConditionType>
class BoundaryConditionTypeInterface{
    public:
           BoundaryConditionTypeInterface(){
            printf("BoundaryConditionTypeInterface CTOR\n");
            copyClassToDevice();
            }
           ~BoundaryConditionTypeInterface(){
            removeClassFromDevice();
            }
            void updateBoundaries(const int _Nx){
              //BoundaryConditionType::updateBoundaries(_Nx);
              static_cast<BoundaryConditionType*>(this)->updateBoundaries(_Nx);
            }

            void copyClassToDevice(){
            #pragma acc enter data copyin(this[0:1])
           }
           void removeClassFromDevice(){
            #pragma acc exit data delete(this[0:1])
           }
};

#endif