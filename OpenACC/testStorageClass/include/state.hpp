#ifndef _STATE_H
#define _STATE_H

#include<vector>
#include<memory>

class State{
    public:
          //Delete copy CTOR
          State(State& _state) = delete;

          //Delete assigment CTOR
          void operator =(const State&) = delete;

          //CTOR
          //totalSize = Nx*Ny(*Nz)
           State(int _totalSize): m_totalSize(_totalSize)
           {
            m_storage.resize(m_totalSize);
           }
          //DTOR
          ~State(){
            m_storage.resize(0);
          }

          //Getter function
          double& getStateAtPoint(int index){
            return(m_storage[index]);
          }
          double* getState(){
            return(m_storage.data());
          }
          int getTotalSize() const {return m_totalSize;}
         // #pragma acc routine seq
          void initializeState(){
            #pragma acc parallel loop
              for (int i = 0; i < m_totalSize; ++i) {
                m_storage[i] = 1;
              }
            }
          //Openacc enabled functions
          void copyClassToDevice(){
            #pragma acc enter data copyin(this[0:1], m_totalSize, m_storage[0:m_totalSize])
          }


    protected:
          //Underground storage array for the data (temperature)
          //Flattened array layout for multidimensional data
          //We envision that the array would be stored as a 3D array [k][j][i]
          //with i being the continuous x directional data
          //Access: For looping over elements use:
          //loop over (k):
          //  loop over (j):
          //    loop over (i):
          //       m_storage[i+j*Nx+k*Nx*Ny]
          std::vector<double> m_storage;
          int m_totalSize;
};

#endif