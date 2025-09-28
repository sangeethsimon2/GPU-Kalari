#pragma once 

#include<iostream>
#include<vector>

//Type that stores the AABB for each object 
struct AABB{
     float minX; float maxX;
     float minY; float maxY;
     float minZ; float maxZ;
     float centroidX{0.}; float centroidY{0.}; float centroidZ{0.};
};

//Type of rectangle object 
struct rectangleObject{
     float originX; float originY; float originZ;
     float width; float height; float depth;
     rectangleObject(float _originX, float _originY, float _width, float _height):
        originX(_originX), originY(_originY), originZ(0.), width(_width), height(_height), depth(0.) {}
};
// Enum class of types of supported objects
enum class objType{
    RECTANGLE = 0
};

// Function to create objects on screen
std::vector<rectangleObject> createObjOnScreen(const size_t, const objType);


//Function to allocate memory and copy objects to device
template<typename T> 
void uploadTodevice(T*, const size_t);

//test Function to download objects from device and deallocate memory 
template<typename T>
void testdownloadToHost(T*, const size_t);

//Function to download objects from device and deallocate memory 
//downloadFromDevice();

// Function to compute the tree
void computeTree(const size_t, int, int);