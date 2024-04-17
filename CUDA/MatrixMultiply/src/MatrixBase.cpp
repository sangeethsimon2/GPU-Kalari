/*Implementations of the base matrix class methods*/
#include "MatrixBase.hpp"

//Method to show the contents of the matrix
void matrix::showMatrix(){
        for(auto i=0;i<_numElementsOnRow*_numElementsOnColumn;i++){
            std::cout<<" i "<<i<<" : ";
            std::cout<<" value = "<<(*_matrixElements)[i]<<"\n";
        }
}
//Method to access the matrix elements at a specific location
double& matrix::operator [](size_t n) {
        return (*_matrixElements)[n];
}
//override
const double& matrix::operator [](size_t n) const { return (*_matrixElements)[n]; }

//An operator overloading to multiply two matrices
 matrix matrix::operator*(const matrix&B) const{
        double sum=0.;
        //check if the matrix dimensions match for performing a multiplication
        if(this->_numElementsOnColumn != B._numElementsOnRow){
            std::cerr<<" The matrix dimensions do not match. Aborting!"<<std::endl;
            abort();
        }
        else{
            matrix C(this->_numElementsOnRow, B._numElementsOnColumn);
            for(auto i=0; i<this->_numElementsOnRow; i++){
                for(auto j=0; j<B._numElementsOnColumn; j++){
                    sum=0.;
                    for(auto k=0; k<this->_numElementsOnRow; k++)
                        sum += (*this)[i*this->_numElementsOnColumn + k]*B[k*B._numElementsOnRow + j];
                    C[i*this->_numElementsOnRow+j] = sum;
                }
            }
        return C;
        }
}
//Method to get matrix elements
 std::vector<double> matrix::getMatrixElements(){
        return(*_matrixElements);
    }

//method to get the total size of the matrix elements
 size_t matrix::getSizeInBytesOfMatrixElements(){
        return( getMatrixElements().size()*sizeof(double) );
    }

//Method to get the total number of elements in the matrix
int matrix::getNumberOfElementsInMatrix(){
        assert(this->_numElementsOnColumn==this->_numElementsOnRow);
        return(this->_numElementsOnColumn);
}

//Method to initialize the matrix to zero
void matrix::initializeMatrixToZero(){
    //std::fill(matrix::getMatrixElements().begin(), matrix::getMatrixElements().end(), 0.);
    for(auto i=0;i<this->_numElementsOnRow*this->_numElementsOnColumn;i++){
            (*this->_matrixElements)[i] = 0.0;
        }
}