#include <iostream>
#include <memory>
#include <vector>
#include <openacc.h>

#include "strategyA.hpp"
#include "strategyB.hpp"

#include "kernel.hpp"
int main(){

   //Create a data field
   //Create a vector of ints
    std::vector<int> vecOfInts;

    // Fill the vector with some data
    for (int i = 0; i < 100; ++i) {
        vecOfInts.push_back(i);
    }
    //Do an unstructured copy here.
     #pragma acc enter data copyin(vecOfInts[0:99])

    //Print the modified vector
    for (int i = 0; i < vecOfInts.size(); ++i)  {
        std::cout << vecOfInts[i] << " ";
    }
    std::cout << std::endl;


    constexpr int DIM = 2;
    //Declare pointers
    //Instantiate Kernel class with StrategyA with its non-type template parameter DIM=2
     std::shared_ptr<Kernel<StrategyA<2>>> m_ptr2Jacobi2D;
    //Instantiate Kernel class with StrategyB with its non-type template parameter DIM=2
     std::shared_ptr<Kernel<StrategyB<2>>> m_ptr2GaussSiedel2D;

    //Instantiate Kernel class with StrategyA with its non-type template parameter DIM=3
     std::shared_ptr<Kernel<StrategyA<3>>> m_ptr2Jacobi3D;
    //Instantiate Kernel class with StrategyB with its non-type template parameter DIM=3
    std::shared_ptr<Kernel<StrategyB<3>>> m_ptr2GaussSiedel3D;

    //Create a Kernel instance
    // This switch could be based on user input DIM value from the parameters
    if(DIM==2){
        m_ptr2Jacobi2D = std::make_shared<Kernel<StrategyA<2>>>(std::make_shared<StrategyA<2>>());
        //Copy pointers and attach so that the subsequent call to updateSolution() can be made on device?
        #pragma acc enter data copyin(m_ptr2Jacobi2D[0:1])
        m_ptr2Jacobi2D->updateSolution(vecOfInts);
        m_ptr2GaussSiedel2D = std::make_shared<Kernel<StrategyB<2>>>(std::make_shared<StrategyB<2>>());
        #pragma acc enter data copyin(m_ptr2GaussSiedel2D[0:1])
        m_ptr2GaussSiedel2D->updateSolution(vecOfInts);
    }
    else if (DIM==3) {
        m_ptr2Jacobi3D = std::make_shared<Kernel<StrategyA<3>>>(std::make_shared<StrategyA<3>>());
        #pragma acc enter data copyin(m_ptr2Jacobi3D[0:1])
        m_ptr2Jacobi3D->updateSolution(vecOfInts);
        m_ptr2GaussSiedel3D = std::make_shared<Kernel<StrategyB<3>>>(std::make_shared<StrategyB<3>>());
        #pragma acc enter data copyin(m_ptr2GaussSiedel3D[0:1])
        m_ptr2GaussSiedel3D->updateSolution(vecOfInts);
    }
    else
        throw std::runtime_error("Invalid value of DIM, unable to initialize kernel\n");

    //Either achieve a an automatic update or do a self update
    #pragma acc update self (vecOfInts[0:99])

    //Print the modified vector
    for (int i = 0; i < vecOfInts.size(); ++i)  {
        std::cout << vecOfInts[i] << " ";
    }
    std::cout << std::endl;

    #pragma acc exit data delete(vecOfInts[0:99])

    #pragma acc exit data delete(m_ptr2Jacobi2D[0:1], m_ptr2GaussSiedel2D[0:1], m_ptr2Jacobi3D[0:1], m_ptr2GaussSiedel3D[0:1])
    return 0;
}