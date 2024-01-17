#ifndef MATOPERONGPUCUDA
#define MATOPERONGPUCUDA
#define tileWidth 16
void printCudaVersionNumber();
void uploadToDevice(const size_t size, const double* src, double* dst);
void downloadToHost(const size_t size, const double* src, double* dst);
void matrixMultiplyOnDevice(const double*, const double*, double*, const int);
#endif