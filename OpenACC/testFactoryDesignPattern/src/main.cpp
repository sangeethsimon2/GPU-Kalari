#include <iostream>
#include <memory>
#include <vector>
#include <openacc.h>

#include "dirichletBoundaryTypeImpl.hpp"
#include "boundaryConditionTypeInterface.hpp"
#include "boundaryCreatorInterface.hpp"
#include "dirichletBoundaryCreator.hpp"
int main(){

   //Create a data field
   constexpr int size = 10;
   //Create a vector of ints
    std::vector<int> vecOfInts;

    // Fill the vector with some data
    for (int i = 0; i < size; ++i) {
        vecOfInts.push_back(i);
    }
    #pragma acc enter data copyin(vecOfInts[0:9])

    //Print the modified vector
    for (int i = 0; i < vecOfInts.size(); ++i)  {
        std::cout << vecOfInts[i] << " ";
    }
    std::cout << std::endl;


    constexpr int DIM = 2;
    //auto m_ptr2BoundaryCreator2D = std::make_shared<BoundaryCreatorInterface<DirichletImpl<2>>>(5);

    auto m_ptr2BoundaryCreator2D = std::make_shared<DirichletBoundaryCreator<DirichletImpl<2>>>(5);
    #pragma acc enter data copyin(m_ptr2BoundaryCreator2D[0:1])


        m_ptr2BoundaryCreator2D->setBoundaryType();
        m_ptr2BoundaryCreator2D->updateBoundaries();
    //This switch could be based on user input DIM value from the parameters
    if(DIM==2){
    }
    // else if (DIM==3) {

    //     m_ptr2BoundaryCreator3D->updateBoundaries();
    //  }
    // else
    //     throw std::runtime_error("Invalid value of DIM, unable to initialize kernel\n");

    #pragma acc update self (vecOfInts[0:9])

    //Print the modified vector
    for (int i = 0; i < vecOfInts.size(); ++i)  {
        std::cout << vecOfInts[i] << " ";
    }
    std::cout << std::endl;

    #pragma acc exit data delete(vecOfInts[0:9])
    #pragma acc exit data delete(m_ptr2BoundaryCreator2D[0:1])


    return 0;
}