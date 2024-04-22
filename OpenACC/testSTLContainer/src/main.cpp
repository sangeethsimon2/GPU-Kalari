#include <iostream>
#include <memory>
#include <vector>

int main(){
    //Create a vector of ints
    std::vector<int> vecOfInts;

    // Fill the vector with some data
    for (int i = 0; i < 1000; ++i) {
        vecOfInts.push_back(i);
    }
    #pragma acc enter data copyin(vecOfInts[0:999])

    //Serial computation
    for (int i = 0; i < 1000; ++i) {
    // Access and modify vector elements safely in parallel
         vecOfInts[i] *= 1;
    }

    // Parallelize a loop over the vector using OpenACC
    #pragma acc data copyout(vecOfInts[0:999])
    {
      #pragma acc parallel loop
      for (int i = 0; i < 1000; ++i) {
        // Access and modify vector elements safely in parallel
        vecOfInts[i] *= 2;
      }
    }
    //#pragma acc update self (vecOfInts[0:999])

    //Print the modified vector
    for (int i = 0; i < vecOfInts.size(); ++i)  {
        std::cout << vecOfInts[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}