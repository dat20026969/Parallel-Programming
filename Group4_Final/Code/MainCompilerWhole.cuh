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



MainCompilerWhole::MainCompilerWhole() {
    srand(time(nullptr));
}



MainCompilerWhole::~MainCompilerWhole() {

    if (d_input_list == NULL)
        return;

    delete[] h_input_list;
    delete[] h_output_L2_list;
    delete[] h_output_L3_list;
    delete[] h_output_L4_list;
    delete[] h_output_L5_list;
    delete[] h_kernel_L1;
    delete[] h_kernel_L3;

    cudaFree(d_input_list);
    cudaFree(d_output_L2_list);
    cudaFree(d_output_L3_list);
    cudaFree(d_output_L4_list);
    cudaFree(d_output_L5_list);

    cudaFree(d_kernel_L1);
    cudaFree(d_kernel_L3);
}




void MainCompilerWhole::copyFile(std::vector<ImageData>& trainData, std::vector<ImageData>& testData) {
    this->trainData = trainData;
    this->testData = testData;

    dataSize = testData.size();
    greatInputXCount = 100;
    greatInputYCount = dataSize / greatInputXCount;
}



void MainCompilerWhole::assignHostMemory() {

    width_L2 = adjustSize(NEW_WIDTH, L1_FILTER_SIZE, 1);
    height_L2 = adjustSize(NEW_HEIGHT, L1_FILTER_SIZE, 1);
    width_L3 = adjustSize(width_L2, L2_POOL_SIZE, 2);
    height_L3 = adjustSize(height_L2, L2_POOL_SIZE, 2);
    width_L4 = adjustSize(width_L3, L3_FILTER_SIZE, 1);
    height_L4 = adjustSize(height_L3, L3_FILTER_SIZE, 1);
    width_L5 = adjustSize(width_L4, L4_POOL_SIZE, 2);
    height_L5 = adjustSize(height_L4, L4_POOL_SIZE, 2);

    std::cout << "\nImage size at layer 1: (" << NEW_WIDTH << ", " << NEW_HEIGHT << ", " << L1_CHANNELS << ")\n";
    std::cout << "Image size at layer 2: (" << width_L2 << ", " << height_L2 << ", " << L2_CHANNELS << ")\n";
    std::cout << "Image size at layer 3: (" << width_L3 << ", " << height_L3 << ", " << L3_CHANNELS << ")\n";
    std::cout << "Image size at layer 4: (" << width_L4 << ", " << height_L4 << ", " << L4_CHANNELS << ")\n";
    std::cout << "Image size at layer 5: (" << width_L5 << ", " << height_L5 << ", " << L5_CHANNELS << ")\n\n";

    h_input_list = new float[NEW_WIDTH * NEW_HEIGHT * dataSize];
    h_output_L2_list = new float[L2_CHANNELS * width_L2 * height_L2 * dataSize];
    h_output_L3_list = new float[L3_CHANNELS * width_L3 * height_L3 * dataSize];
    h_output_L4_list = new float[L4_CHANNELS * width_L4 * height_L4 * dataSize];
    h_output_L5_list = new float[L5_CHANNELS * width_L5 * height_L5 * dataSize];
}



void MainCompilerWhole::assignDeviceMemory() {

    CHECK(cudaMalloc(&d_input_list, NEW_WIDTH * NEW_HEIGHT * dataSize * sizeof(float)));

    CHECK(cudaMalloc(&d_output_L2_list, width_L2 * height_L2 * L2_CHANNELS * dataSize * sizeof(float)));
    CHECK(cudaMalloc(&d_output_L3_list, width_L3 * height_L3 * L3_CHANNELS * dataSize * sizeof(float)));
    CHECK(cudaMalloc(&d_output_L4_list, width_L4 * height_L4 * L4_CHANNELS * dataSize * sizeof(float)));
    CHECK(cudaMalloc(&d_output_L5_list, width_L5 * height_L5 * L5_CHANNELS * dataSize * sizeof(float)));

    int L1_kernelSize = L1_FILTERS * L1_FILTER_SIZE * L1_FILTER_SIZE;
    CHECK(cudaMalloc(&d_kernel_L1, L1_kernelSize * sizeof(float)));
    CHECK(cudaMemcpy(d_kernel_L1, h_kernel_L1, L1_kernelSize * sizeof(float), cudaMemcpyHostToDevice));

    int L3_kernelSize = L3_FILTERS * L3_FILTER_SIZE * L3_FILTER_SIZE * L1_FILTERS;
    CHECK(cudaMalloc(&d_kernel_L3, L3_kernelSize * sizeof(float)));
    CHECK(cudaMemcpy(d_kernel_L3, h_kernel_L3, L3_kernelSize * sizeof(float), cudaMemcpyHostToDevice));
}



void MainCompilerWhole::handleRunOnHostWithBiggerImage() {

    runConvolutionByHostWithLargerImage(h_input_list, NEW_WIDTH, NEW_HEIGHT,
        h_output_L2_list, width_L2, height_L2,
        h_kernel_L1, L1_FILTER_SIZE, L1_FILTER_SIZE, L1_CHANNELS, L2_CHANNELS,
        greatInputXCount, greatInputYCount);

    ReLUByHost(h_output_L2_list, h_output_L2_list, width_L2 * greatInputXCount, height_L2 * greatInputYCount, L2_CHANNELS);

    //runMaxPoolingByHost(h_output_L2_list, width_L2, height_L2,
    //    h_output_L3_list, width_L3 * greatInputXCount, height_L3 * greatInputYCount, L2_POOL_SIZE, L2_POOL_SIZE, L3_CHANNELS);


    //runConvolutionByHostWithLargerImage(h_output_L3_list, width_L3, height_L3,
    //    h_output_L4_list, width_L4, height_L4,
    //    h_kernel_L3, L3_FILTER_SIZE, L3_FILTER_SIZE, L3_CHANNELS, L4_CHANNELS,
    //    greatInputXCount, greatInputYCount);

    //ReLUByHost(h_output_L4_list, h_output_L4_list, width_L4 * greatInputXCount, height_L4 * greatInputYCount, L4_CHANNELS);

    //runMaxPoolingByHost(h_output_L4_list, width_L4, height_L4,
    //    h_output_L5_list, width_L5 * greatInputXCount, height_L5 * greatInputYCount, L4_POOL_SIZE, L4_POOL_SIZE, L5_CHANNELS);
}



// Có stream
// Sử dụng kernel 2
void MainCompilerWhole::handleRunOnDeviceWithBiggerImage(int& blockSizeVal1, int blockSizeVal3,
    dim3& blockSize1, dim3& gridSize1, dim3& blockSize3, dim3& gridSize3) {

    // Truyền input vào
    CHECK(cudaMemcpy(d_input_list, h_input_list, NEW_WIDTH * NEW_HEIGHT * dataSize * sizeof(float), cudaMemcpyHostToDevice));


    // ----------------------------- Lớp 1 ----------------------------- 

    size_t sharedSize1;
    sharedSize1 = (blockSizeVal1 + L1_FILTER_SIZE - 1) *
        (blockSizeVal1 + L1_FILTER_SIZE - 1) * L2_CHANNELS * sizeof(float);

    optimizedConvolutionWithBiggerImage <<< gridSize1, blockSize1, sharedSize1 >>> (
        d_input_list, NEW_WIDTH, NEW_HEIGHT,
        d_output_L2_list, width_L2, height_L2,
        d_kernel_L1, L1_FILTER_SIZE, L1_FILTER_SIZE, L1_CHANNELS, L2_CHANNELS,
        greatInputXCount, greatInputYCount
    );

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    ReLUByDevice <<< gridSize1, blockSize1 >>> (
        d_output_L2_list, d_output_L2_list, width_L2 * greatInputXCount, height_L2 * greatInputYCount, L2_CHANNELS);

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());


    // -----------------------------  Lớp 2 ----------------------------- 
    maxPoolingByDevice <<< gridSize1, blockSize1 >>> (
        d_output_L2_list, width_L2 * greatInputXCount, height_L2 * greatInputYCount,
        d_output_L3_list, width_L3 * greatInputXCount, height_L3 * greatInputYCount,
        L2_POOL_SIZE, L2_POOL_SIZE, L3_CHANNELS, 2);

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());


    // ----------------------------- Lớp 3 ----------------------------- 
    size_t sharedSize3;
    sharedSize3 = (blockSizeVal3 + L3_FILTER_SIZE - 1) *
        (blockSizeVal3 + L3_FILTER_SIZE - 1) * L4_CHANNELS * sizeof(float);

    optimizedConvolutionWithBiggerImage <<< gridSize3, blockSize3, sharedSize3 >>> (
        d_output_L3_list, width_L3, height_L3,
        d_output_L4_list, width_L4, height_L4,
        d_kernel_L3, L3_FILTER_SIZE, L3_FILTER_SIZE, L3_CHANNELS, L4_CHANNELS,
        greatInputXCount, greatInputYCount
        );

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    ReLUByDevice <<< gridSize3, blockSize3 >>> (
        d_output_L4_list, d_output_L4_list, width_L4 * greatInputXCount, height_L4 * greatInputYCount, L4_CHANNELS);

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());


    // ----------------------------- Lớp 4 ----------------------------- 
    maxPoolingByDevice <<< gridSize3, blockSize3 >>> (
        d_output_L4_list, width_L4 * greatInputXCount, height_L4 * greatInputYCount,
        d_output_L5_list, width_L5 * greatInputXCount, height_L5 * greatInputYCount,
        L4_POOL_SIZE, L4_POOL_SIZE, L4_CHANNELS, 2);

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Lấy kết quả cuối cùng (Layer 5)
    CHECK(cudaMemcpy(h_output_L5_list, d_output_L5_list,
        width_L5 * height_L5 * L5_CHANNELS * dataSize * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(h_output_L4_list, d_output_L4_list,
        width_L4 * height_L4 * L4_CHANNELS * dataSize * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(h_output_L3_list, d_output_L3_list,
        width_L3 * height_L3 * L3_CHANNELS * dataSize * sizeof(float), cudaMemcpyDeviceToHost));


    // ----------------------------------------------------------------
}



void MainCompilerWhole::assignInputData() {

    // std::cout << "Total size: " << NEW_WIDTH * NEW_HEIGHT * dataSize << "\n";

    for (int id = 0; id < dataSize; id++) {

        int shift_i = id / greatInputXCount * NEW_HEIGHT;
        int shift_j = id % greatInputXCount * NEW_WIDTH;

        for (int i = 0; i < NEW_HEIGHT; i++) {
            for (int j = 0; j < NEW_WIDTH; j++) {
                //std::cout << "Curr id at ["
                //    << shift_i << " + " << i << ", " << j << " + " << shift_j
                //    << "]: "
                //    << (shift_i + i) * greatInputXCount * NEW_WIDTH + j + shift_j << "\n";
                h_input_list[(shift_i + i) * greatInputXCount * NEW_WIDTH + j + shift_j]
                    = testData[id].image[i * NEW_WIDTH + j];
            }
        }
    }
}



float MainCompilerWhole::runWithBiggerImage(bool onDevice, int blockSizeVal1, int blockSizeVal3) {

    std::cout << "Start running\n";

    GpuTimer timer;

    if (onDevice) {
        dim3 blockSize1(blockSizeVal1, blockSizeVal1);
        dim3 gridSize1((NEW_WIDTH * greatInputXCount + blockSize1.x - 1) / blockSize1.x, (NEW_HEIGHT * greatInputYCount + blockSize1.y - 1) / blockSize1.y);

        dim3 blockSize3(blockSizeVal3, blockSizeVal3);
        dim3 gridSize3((width_L3 * greatInputXCount + blockSize3.x - 1) / blockSize3.x, (height_L3 * greatInputYCount + blockSize3.y - 1) / blockSize3.y);

        timer.Start();

        handleRunOnDeviceWithBiggerImage(blockSizeVal1, blockSizeVal3,
            blockSize1, gridSize1, blockSize3, gridSize3);
    }
    else {
        timer.Start();
        handleRunOnHostWithBiggerImage();
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    return timer.Elapsed();
}



void MainCompilerWhole::loadKernel() {

    h_kernel_L1 = new float[L1_FILTERS * L1_FILTER_SIZE * L1_FILTER_SIZE];
    h_kernel_L3 = new float[L3_FILTERS * L3_FILTER_SIZE * L3_FILTER_SIZE * L1_FILTERS];


    // Kernel 1
    std::string file_name = "Images/Kernel1.txt";
    std::ifstream k1_file(file_name);
    if (k1_file.is_open()) {
        for (int nc = 0; nc < L2_CHANNELS; nc++) {
            for (int i = 0; i < L1_FILTER_SIZE; i++) {
                for (int j = 0; j < L1_FILTER_SIZE; j++) {
                    unsigned int kernel_pos = nc * L1_FILTER_SIZE * L1_FILTER_SIZE + j * L1_FILTER_SIZE + i;
                    float temp;
                    k1_file >> temp;
                    h_kernel_L1[kernel_pos] = temp;
                }
            }
        }
    }
    else {
        std::cerr << "Unable to read such a file " << file_name << "\n";
    }
    k1_file.close();



    file_name = "Images/Kernel3.txt";
    std::ifstream k3_file(file_name);
    if (k3_file.is_open()) {
        for (int oc = 0; oc < L3_CHANNELS; oc++) {
            for (int nc = 0; nc < L4_CHANNELS; nc++) {
                for (int i = 0; i < L1_FILTER_SIZE; i++) {
                    for (int j = 0; j < L1_FILTER_SIZE; j++) {
                        unsigned int kernel_pos = (nc * L1_FILTER_SIZE * L1_FILTER_SIZE * L3_CHANNELS) +
                            (j * L1_FILTER_SIZE * L3_CHANNELS) +
                            (i * L3_CHANNELS) + oc;
                        float temp;
                        k3_file >> temp;
                        h_kernel_L3[kernel_pos] = temp;
                    }
                }
            }
        }
    }
    else {
        std::cerr << "Unable to write such a file " << file_name << "\n";
    }
    k3_file.close();
}


#endif