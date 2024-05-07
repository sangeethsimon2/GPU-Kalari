#include <iostream>
#include <memory>
#include <vector>
#include <openacc.h>
int main(){
    // Create a unique pointer to an STL vector of ints
    std::shared_ptr<std::vector<int>> vec_ptr = std::make_shared<std::vector<int>>();

    constexpr int SIZE=100;
    // Fill the vector with some data
    for (int i = 0; i < SIZE; ++i) {
        vec_ptr->push_back(i);
    }
    #pragma acc enter data copyin(vec_ptr[0:1], vec_ptr[0:SIZE])


    // Parallelize a loop over the vector using OpenACC
    #pragma acc data copyout(vec_ptr[0:SIZE])
    {
      #pragma acc parallel loop
      for (int i = 0; i < vec_ptr->size(); ++i) {
        printf(" The host, device flags are %d, %d \n", acc_on_device(acc_device_host), acc_on_device(acc_device_nvidia));
        // Access and modify vector elements safely in parallel
        (*vec_ptr)[i] *= 2;
      }
    }

    #pragma acc update self (vec_ptr[0:SIZE])
    //Print the modified vector from host
    for (int i = 0; i < vec_ptr->size(); ++i)  {
        std::cout << (*vec_ptr)[i] << " ";
    }
    std::cout << std::endl;

    #pragma acc exit data delete (vec_ptr[0:SIZE], vec_ptr[0:1])
    return 0;
}