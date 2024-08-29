#ifndef PROPERTIES_
#define PROPERTIES_


#include "cuda_runtime.h"

#include "stdio.h"
#include <algorithm>



constexpr int IMAGE_WIDTH = 28;
constexpr int IMAGE_HEIGHT = 28;

constexpr int KERNEL_WIDTH = 5;
constexpr int KERNEL_HEIGHT = 5;

constexpr int PADDING_WIDTH = 2;
constexpr int PADDING_HEIGHT = 2;

constexpr int TILE_WIDTH = 16;
constexpr int TILE_HEIGHT = 16;

constexpr int NEW_WIDTH = 32;  // Example value, adjust as necessary
constexpr int NEW_HEIGHT = 32; // Example value, adjust as necessary

const int L1_FILTER_SIZE = 5;
constexpr int L1_FILTERS = 6;
constexpr int L2_POOL_SIZE = 2;
constexpr int L3_FILTERS = 16;
constexpr int L3_FILTER_SIZE = 5;
constexpr int L4_POOL_SIZE = 2;
constexpr int L6_NEURONS = 120;
constexpr int L7_NEURONS = 84;
constexpr int OUTPUT_CLASSES = 10;

constexpr int L1_CHANNELS = 1;
constexpr int L2_CHANNELS = 6;
constexpr int L3_CHANNELS = 6;
constexpr int L4_CHANNELS = 16;
constexpr int L5_CHANNELS = 16;



#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}



struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop() {
        cudaEventRecord(stop, 0);
    }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};



int adjustSize(int oldSize, int kernelSize, int stride);
void assignMemory(float* assigned, float* assigning, int width, int height, int channels);
float checkAccuracy(float* first, float* second, int width, int height, int channels);




#endif