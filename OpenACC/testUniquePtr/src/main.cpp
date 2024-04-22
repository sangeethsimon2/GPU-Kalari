#include <iostream>
#include <memory>
#include <vector>

int main(){
    // Create a unique pointer to an STL vector of ints
    std::shared_ptr<std::vector<int>> vec_ptr = std::make_shared<std::vector<int>>();

    // Fill the vector with some data
    for (int i = 0; i < 1000; ++i) {
        vec_ptr->push_back(i);
    }
    #pragma acc enter data copyin(vec_ptr[0:1], vec_ptr[0:999])


    // Parallelize a loop over the vector using OpenACC
    #pragma acc data copyout(vec_ptr[0:999])
    {
      #pragma acc parallel loop
      for (int i = 0; i < 1000; ++i) {
        // Access and modify vector elements safely in parallel
        (*vec_ptr)[i] *= 2;
      }
    }
    //Print the modified vector
    for (int i = 0; i < vec_ptr->size(); ++i)  {
        std::cout << (*vec_ptr)[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}