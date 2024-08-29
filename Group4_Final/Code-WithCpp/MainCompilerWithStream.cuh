#ifndef MAIN_COMPILER_WITH_STREAM_
#define MAIN_COMPILER_WITH_STREAM_



#include "cuda_runtime.h"

#include "Properties.cuh"
#include "ImageData.h"
#include "FileHandler.h"
#include "ConvolutionHost.h"
#include "ConvolutionDevice.cuh"
#include "MainCompiler.cuh"

#include <string>
#include <vector>
#include <fstream>
#include <random>




class MainCompilerWithStream {

private:

    const int streamCount = 128;

    float** h_input_list = NULL;  // Layer 1
    float** h_output_L5_list = NULL;

    float** h_kernel_L1_list = NULL;
    float** h_kernel_L3_list = NULL;


    float** d_input_list = NULL;  // Layer 1
    float** d_output_L2_list = NULL;
    float** d_output_L3_list = NULL;
    float** d_output_L4_list = NULL;
    float** d_output_L5_list = NULL;

    float** d_kernel_L1_list = NULL;
    float** d_kernel_L3_list = NULL;


    int width_L2 = 0;
    int height_L2 = 0;
    int width_L3 = 0;
    int height_L3 = 0;
    int width_L4 = 0;
    int height_L4 = 0;
    int width_L5 = 0;
    int height_L5 = 0;


    cudaStream_t* stream_list = NULL;


    // Image list
    std::vector<ImageData> trainData, testData;

    void handleRunOnDeviceWithStream(int& image_id, int& blockSizeVal1, int blockSizeVal3,
        dim3& blockSize1, dim3& gridSize1, dim3& blockSize3, dim3& gridSize3, int streamId);



public:
    MainCompilerWithStream();
    ~MainCompilerWithStream();

    void copyFile(std::vector<ImageData>& trainData, std::vector<ImageData>& testData);

    void assignHostMemory();
    void assignDeviceMemory();
    float runWithStream(int blockSizeVal1 = 16, int blockSizeVal3 = 8);

    void loadKernel();
};


#endif