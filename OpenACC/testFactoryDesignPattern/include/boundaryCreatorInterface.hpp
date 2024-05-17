/* Virtual base class that is tasked with instantiating the correct
boundary condition implementation through the 'factory method' pattern
and using it to update the boundaries.
The interface of this class interacts with the main*/

#ifndef _BOUNDARYUPDATERINTERFACE_H
#define _BOUNDARYUPDATERINTERFACE_H
#include <cstdio>
#include <memory>
#include "boundaryConditionTypeInterface.hpp"
#include "dirichletBoundaryTypeImpl.hpp"
template<typename BoundaryConditionCreatorType, typename BoundaryConditionType>
class BoundaryCreatorInterface{
    public:
           //CTOR
           BoundaryCreatorInterface(int _Nx): m_Nx(_Nx){
            printf("BoundaryCreator CTOR\n");
            copyClassToDevice();
              }

           ~BoundaryCreatorInterface(){
            removeClassFromDevice();
           }

           //Method to call the update method generated from the factory method
           void updateBoundaries(){
            printf("Call updateBoundaries in BoundaryCreator\n");
            m_BCType->updateBoundaries(m_Nx);
           }

           void setBoundaryType(){
            printf("Call setBoundaryType in BoundaryCreator\n");
            m_BCType = static_cast<BoundaryConditionCreatorType*>(this)->createBoundaryType();
           }

           void copyClassToDevice(){
            #pragma acc enter data copyin(this[0:1])
           }
           void removeClassFromDevice(){
            #pragma acc exit data delete(this[0:1])
           }


    protected:
          std::shared_ptr<BoundaryConditionTypeInterface<BoundaryConditionType>> m_BCType;
          int m_Nx;
};

#endif