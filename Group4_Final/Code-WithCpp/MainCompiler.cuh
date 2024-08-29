#ifndef MAIN_COMPILER_
#define MAIN_COMPILER_



#include "cuda_runtime.h"

#include "Properties.cuh"
#include "ImageData.h"
#include "FileHandler.h"
#include "ConvolutionHost.h"
#include "ConvolutionDevice.cuh"

#include <string>
#include <vector>
#include <fstream>
#include <random>




class MainCompiler {

private:

    int image_id = 0;
    int compiling_type = 0;
    float compiling_time = 0.0f;

    float* h_input = NULL;  // Layer 1
    float* h_output_L2 = NULL;
    float* h_output_L3 = NULL;
    float* h_output_L4 = NULL;
    float* h_output_L5 = NULL;

    float* h_kernel_L1 = NULL;
    float* h_kernel_L3 = NULL;


    float* d_input = NULL;  // Layer 1
    float* d_output_L2 = NULL;
    float* d_output_L3 = NULL;
    float* d_output_L4 = NULL;
    float* d_output_L5 = NULL;

    float* d_kernel_L1 = NULL;
    float* d_kernel_L3 = NULL;



    int width_L2 = 0;
    int height_L2 = 0;
    int width_L3 = 0;
    int height_L3 = 0;
    int width_L4 = 0;
    int height_L4 = 0;
    int width_L5 = 0;
    int height_L5 = 0;

    void handleRunOnHost(int& image_id);
    void handleRunOnDevice(int& image_id, int& compiling_type, int& blockSizeVal1, int blockSizeVal3,
        dim3& blockSize1, dim3& gridSize1, dim3& blockSize3, dim3& gridSize3);


public:

    // Image list
    std::vector<ImageData> trainData, testData;

    MainCompiler();
    ~MainCompiler();

    void readFile(std::string trainImagesFilename = "train-images-idx3-ubyte",
        std::string testImagesFilename = "t10k-images-idx3-ubyte");

    void assignHostMemory();
    void assignDeviceMemory();
    float runOnHost(int image_id = 0);
    float runOnDevice(int image_id = 0, int compiling_type = 1, int blockSize1 = 16, int blockSize3 = 8);
    void runAll(int image_id = 0, int blockSize1 = 16, int blockSize3 = 8);

    void saveRawText();
    void saveKernel();
    void loadKernel();
    void generateKernel();
};



#endif