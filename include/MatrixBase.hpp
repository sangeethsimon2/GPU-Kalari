#include<vector>
#include<random>
#include<memory>
#include<iostream>
#include<stdio.h>
#include<cassert>

// Objectives of this class
// This class implements a matrix
// This class has to 'know' the number of elements (row*column) it has to contain
// This class has to then allocate the array on heap and own this allocation
// It uses RAII through a shared pointer to the allocated memory
class matrix{
    public:
    //Default constructor
    matrix(){
        _numElementsOnRow = 0;
        _numElementsOnColumn = 0;
        _matrixElements = std::make_shared<std::vector<double>>();
    };

    //Constructor with sizes specified
    matrix(int row, int column){
        _numElementsOnRow = row;
        _numElementsOnColumn = column;
        _matrixElements = std::make_shared<std::vector<double>>(_numElementsOnRow*_numElementsOnColumn);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for(auto i=0;i<_numElementsOnRow*_numElementsOnColumn;i++){
            double randomValue = distribution(gen);
            (*_matrixElements)[i] = randomValue;
        }
    };



    //Copy constructor - only if you need a deep copy
    matrix(const matrix& _otherMatrix){
        _numElementsOnRow = _otherMatrix._numElementsOnRow;
        _numElementsOnColumn = _otherMatrix._numElementsOnColumn;
        _matrixElements = std::make_shared<std::vector<double>>(*_otherMatrix._matrixElements.get());
    }
    //Copy assignment operator
    matrix& operator=(const matrix& _otherMatrix){
        if (_otherMatrix._matrixElements) {
                _matrixElements = std::make_shared<std::vector<double>>(*_otherMatrix._matrixElements.get());
            }
        return *this;
    }

    //Method to print the matrix
    void showMatrix();
    //Operator overload for []
    double& operator [](size_t);
    const double& operator [](size_t) const;

    //Operator overload for * to perform matrix multiplication
    matrix operator*(const matrix&) const;

    //Method to return the data held by shared ptr
    std::vector<double> getMatrixElements();

    //Method to return the size of the data (in bytes) held by the shared ptr
    size_t getSizeInBytesOfMatrixElements();

    //Method to return the number of elements in the matrix
    int getNumberOfElementsInMatrix();

    //Method to initialize the elements of the matrix to 0
    void initializeMatrixToZero();

   private:
        int _numElementsOnRow, _numElementsOnColumn;
        std::shared_ptr<std::vector<double>> _matrixElements;
};