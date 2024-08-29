#ifndef MAIN_COMPILER_WHOLE_
#define MAIN_COMPILER_WHOLE_



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




class MainCompilerWhole {

private:

    int image_id = 0;
    int compiling_type = 0;
    float compiling_time = 0.0f;

    float* h_input_list = NULL;  // Layer 1
    float* h_output_L2_list = NULL;
    float* h_output_L3_list = NULL;
    float* h_output_L4_list = NULL;
    float* h_output_L5_list = NULL;

    float* h_kernel_L1 = NULL;
    float* h_kernel_L3 = NULL;


    float* d_input_list = NULL;  // Layer 1
    float* d_output_L2_list = NULL;
    float* d_output_L3_list = NULL;
    float* d_output_L4_list = NULL;
    float* d_output_L5_list = NULL;

    float* d_kernel_L1 = NULL;
    float* d_kernel_L3 = NULL;

    size_t dataSize = 0;
    size_t greatInputXCount = 0;
    size_t greatInputYCount = 0;



    int width_L2 = 0;
    int height_L2 = 0;
    int width_L3 = 0;
    int height_L3 = 0;
    int width_L4 = 0;
    int height_L4 = 0;
    int width_L5 = 0;
    int height_L5 = 0;

    void handleRunOnDeviceWithBiggerImage(int& blockSizeVal1, int blockSizeVal3,
        dim3& blockSize1, dim3& gridSize1, dim3& blockSize3, dim3& gridSize3);

    void handleRunOnHostWithBiggerImage();


public:

    // Image list
    std::vector<ImageData> trainData, testData;

    MainCompilerWhole();
    ~MainCompilerWhole();

    void copyFile(std::vector<ImageData>& trainData, std::vector<ImageData>& testData);

    void assignHostMemory();
    void assignDeviceMemory();
    void assignInputData();
    float runWithBiggerImage(bool onDevice = false, int blockSizeVal1 = 16, int blockSizeVal3 = 8);

    void printFirstResult() {
        //for (int c = 0; c < L5_CHANNELS; c++) {
        //    std::cout << "Channel " << c << "\n";
        //    for (int i = 0; i < height_L5; i++) {
        //        for (int j = 0; j < width_L5; j++) {
        //            printf("%.5f, ", h_output_L5_list[(c * height_L5 * greatInputYCount + i + 25) * width_L5 * greatInputXCount + j + 25]);
        //        }
        //        std::cout << "\n";
        //    }
        //    std::cout << "\n";
        //}

        //int imageId = 0;
        //int shift_i = imageId / greatInputXCount * height_L2;
        //int shift_j = imageId % greatInputXCount * width_L2;

        //for (int c = 0; c < 1; c++) {
        //    std::cout << "Channel " << c << "\n";
        //    for (int i = 0; i < 100; i++) {
        //        for (int j = 0; j < 100; j++) {
        //            printf("%.5f, ", h_output_L2_list[(c * height_L2 * greatInputYCount + i + shift_i) * width_L2 * greatInputXCount + j + shift_j]);
        //        }
        //        std::cout << "\n";
        //    }
        //    std::cout << "\n";
        //}

        std::ofstream f;
        f.open("TEMP.csv");

        if (f.is_open()) {

            int size = 256;
            int width = 10;
            int height = 10;

            int imageId = 0;
            int shift_i = imageId / greatInputXCount * height;
            int shift_j = imageId % greatInputXCount * width;

            for (int c = 0; c < 1; c++) {
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        f << h_output_L4_list[(c * height * greatInputYCount + i + shift_i) * width * greatInputXCount + j + shift_j];
                        if (j < size - 1)
                            f << ",";
                    }
                    f << "\n";
                }
                f << "\n";
            }
        }

        else {
            std::cerr << "Unable to open file.\n";
        }
    }

    void loadKernel();
};



#endif