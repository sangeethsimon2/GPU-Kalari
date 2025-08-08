#pragma once 

#include<iostream>
#include<vector>

//Type that stores the AABB for each object 
struct AABB{
     float minX; float minY;
     float maxX; float maxY;
};

//Type of rectangle object 
struct rectangleObject{
     float originX; float originY;
     float width; float height;
     rectangleObject(float _originX, float _originY, float _width, float _height):
        originX(_originX), originY(_originY), width(_width), height(_height) {}
};
// Enum class of types of supported objects
enum class objType{
    RECTANGLE = 0
};

// Function to create objects on screen
std::vector<rectangleObject> createObjOnScreen(const size_t, const objType);


//Function to allocate memory and copy objects to device 
void uploadTodevice(rectangleObject*, const size_t);

//test Function to download objects from device and deallocate memory 
void testdownloadToHost(rectangleObject*, const size_t);

//Function to download objects from device and deallocate memory 
//downloadFromDevice();

// Function to compute the tree
void computeTree(const size_t, int, int);