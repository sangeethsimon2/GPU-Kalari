#include "AABB.h"


//ONLY .cpp compiles here
std::vector<rectangleObject> createObjOnScreen(const size_t numObj, const objType _objType){
    if(_objType == objType::RECTANGLE){
        std::vector<rectangleObject> objArray; 
        objArray.reserve(numObj);

        //TODO: Specific for 10 rectangle case. To be generalized
        objArray.push_back(rectangleObject(0., 0., 10., 5.));
        objArray.push_back(rectangleObject(5., 2., 8., 6.));
        objArray.push_back(rectangleObject(-3., 1., 4., 7.));
        objArray.push_back(rectangleObject(15., 10., 2., 2.));
        objArray.push_back(rectangleObject(1., 1., 1., 1.));
        objArray.push_back(rectangleObject(100., 50., 20., 10.));
        objArray.push_back(rectangleObject(50., 100., 10., 20.));
        objArray.push_back(rectangleObject(0., 0., 0., 0.)); // Zero-sized rectangle
        objArray.push_back(rectangleObject(7., 7., 3., 3.));
        objArray.push_back(rectangleObject(-10., -5.0, 12.0, 8.0));
        
        return objArray;
    }
    else{
        std::cerr<<"Object Type is not implemented"<<std::endl;
        exit(1);
    }
}


