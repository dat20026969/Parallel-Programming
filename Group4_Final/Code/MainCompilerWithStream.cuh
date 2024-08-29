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


MainCompilerWithStream::MainCompilerWithStream() {
    srand(time(nullptr));
}



MainCompilerWithStream::~MainCompilerWithStream() {

    if (d_input_list == NULL)
        return;

    for (int i = 0; i < streamCount; i++) {
        delete[] h_input_list[i];
        delete[] h_output_L5_list[i];
        delete[] h_kernel_L1_list[i];
        delete[] h_kernel_L3_list[i];

        cudaFree(d_input_list[i]);
        cudaFree(d_output_L2_list[i]);
        cudaFree(d_output_L3_list[i]);
        cudaFree(d_output_L4_list[i]);
        cudaFree(d_output_L5_list[i]);

        cudaFree(d_kernel_L1_list[i]);
        cudaFree(d_kernel_L3_list[i]);
    }
    delete[] h_input_list;
    delete[] h_output_L5_list;
    delete[] h_kernel_L1_list;
    delete[] h_kernel_L3_list;

    delete[] d_input_list;
    delete[] d_output_L2_list;
    delete[] d_output_L3_list;
    delete[] d_output_L4_list;
    delete[] d_output_L5_list;

    delete[] d_kernel_L1_list;
    delete[] d_kernel_L3_list;

    delete[] stream_list;
}




void MainCompilerWithStream::copyFile(std::vector<ImageData>& trainData, std::vector<ImageData>& testData) {
    this->trainData = trainData;
    this->testData = testData;
}



void MainCompilerWithStream::assignHostMemory() {

    h_input_list = new float* [streamCount];
    for (int i = 0; i < streamCount; i++)
        h_input_list[i] = new float[NEW_WIDTH * NEW_HEIGHT];

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

    h_output_L5_list = new float* [streamCount];
    for (int i = 0; i < streamCount; i++)
        h_output_L5_list[i] = new float[L5_CHANNELS * width_L5 * height_L5];

    stream_list = new cudaStream_t[streamCount];
}



void MainCompilerWithStream::assignDeviceMemory() {

    d_input_list = new float* [streamCount];
    d_output_L2_list = new float* [streamCount];
    d_output_L3_list = new float* [streamCount];
    d_output_L4_list = new float* [streamCount];
    d_output_L5_list = new float* [streamCount];

    d_kernel_L1_list = new float* [streamCount];
    d_kernel_L3_list = new float* [streamCount];

    for (int i = 0; i < streamCount; i++) {

        CHECK(cudaMalloc(&d_input_list[i], NEW_WIDTH * NEW_HEIGHT * sizeof(float)));

        CHECK(cudaMalloc(&d_output_L2_list[i], width_L2 * height_L2 * L2_CHANNELS * sizeof(float)));
        CHECK(cudaMalloc(&d_output_L3_list[i], width_L3 * height_L3 * L3_CHANNELS * sizeof(float)));
        CHECK(cudaMalloc(&d_output_L4_list[i], width_L4 * height_L4 * L4_CHANNELS * sizeof(float)));
        CHECK(cudaMalloc(&d_output_L5_list[i], width_L5 * height_L5 * L5_CHANNELS * sizeof(float)));

        int L1_kernelSize = L1_FILTERS * L1_FILTER_SIZE * L1_FILTER_SIZE;
        CHECK(cudaMalloc(&d_kernel_L1_list[i], L1_kernelSize * sizeof(float)));
        CHECK(cudaMemcpy(d_kernel_L1_list[i], h_kernel_L1_list[i], L1_kernelSize * sizeof(float), cudaMemcpyHostToDevice));

        int L3_kernelSize = L3_FILTERS * L3_FILTER_SIZE * L3_FILTER_SIZE * L1_FILTERS;
        CHECK(cudaMalloc(&d_kernel_L3_list[i], L3_kernelSize * sizeof(float)));
        CHECK(cudaMemcpy(d_kernel_L3_list[i], h_kernel_L3_list[i], L3_kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    }
}




// Có stream
// Sử dụng kernel 2
void MainCompilerWithStream::handleRunOnDeviceWithStream(int& image_id, int& blockSizeVal1, int blockSizeVal3,
    dim3& blockSize1, dim3& gridSize1, dim3& blockSize3, dim3& gridSize3, int id) {

    // Before
    for (unsigned int i = 0; i < NEW_HEIGHT; i++) {
        for (unsigned int j = 0; j < NEW_WIDTH; j++) {
            h_input_list[id][i * NEW_WIDTH + j] = testData[image_id].image[i * NEW_WIDTH + j];
        }
    }

    cudaStreamCreate(&stream_list[id]);

    // Truyền input vào
    CHECK(cudaMemcpyAsync(d_input_list[id], h_input_list[id],
        NEW_WIDTH * NEW_HEIGHT * sizeof(float), cudaMemcpyHostToDevice, stream_list[id]));

    // ----------------------------- Lớp 1 ----------------------------- 
    size_t sharedSize1;
    sharedSize1 = (blockSizeVal1 + L1_FILTER_SIZE - 1) *
        (blockSizeVal1 + L1_FILTER_SIZE - 1) * L2_CHANNELS * sizeof(float);

    optimizedConvolutionByDevice << < gridSize1, blockSize1, sharedSize1, stream_list[id] >> > (
        d_input_list[id], NEW_WIDTH, NEW_HEIGHT,
        d_output_L2_list[id], width_L2, height_L2,
        d_kernel_L1_list[id], L1_FILTER_SIZE, L1_FILTER_SIZE, L1_CHANNELS, L2_CHANNELS
        );

    cudaStreamSynchronize(stream_list[id]);
    CHECK(cudaGetLastError());

    ReLUByDevice << < gridSize1, blockSize1, 0, stream_list[id] >> > (
        d_output_L2_list[id], d_output_L2_list[id], width_L2, height_L2, L2_CHANNELS);

    cudaStreamSynchronize(stream_list[id]);
    CHECK(cudaGetLastError());


    // -----------------------------  Lớp 2 ----------------------------- 
    maxPoolingByDevice << < gridSize1, blockSize1, 0, stream_list[id] >> > (
        d_output_L2_list[id], width_L2, height_L2,
        d_output_L3_list[id], width_L3, height_L3,
        L2_POOL_SIZE, L2_POOL_SIZE, L3_CHANNELS, 2);

    cudaStreamSynchronize(stream_list[id]);
    CHECK(cudaGetLastError());


    // ----------------------------- Lớp 3 ----------------------------- 
    size_t sharedSize3;
    sharedSize3 = (blockSizeVal3 + L3_FILTER_SIZE - 1) *
        (blockSizeVal3 + L3_FILTER_SIZE - 1) * L4_CHANNELS * sizeof(float);

    optimizedConvolutionByDevice << < gridSize3, blockSize3, sharedSize3, stream_list[id] >> > (
        d_output_L3_list[id], width_L3, height_L3,
        d_output_L4_list[id], width_L4, height_L4,
        d_kernel_L3_list[id], L3_FILTER_SIZE, L3_FILTER_SIZE, L3_CHANNELS, L4_CHANNELS
        );

    cudaStreamSynchronize(stream_list[id]);
    CHECK(cudaGetLastError());

    ReLUByDevice << < gridSize3, blockSize3, 0, stream_list[id] >> > (
        d_output_L4_list[id], d_output_L4_list[id], width_L4, height_L4, L4_CHANNELS);

    cudaStreamSynchronize(stream_list[id]);
    CHECK(cudaGetLastError());


    // ----------------------------- Lớp 4 ----------------------------- 
    maxPoolingByDevice << < gridSize3, blockSize3, 0, stream_list[id] >> > (
        d_output_L4_list[id], width_L4, height_L4,
        d_output_L5_list[id], width_L5, height_L5,
        L4_POOL_SIZE, L4_POOL_SIZE, L4_CHANNELS, 2);

    cudaStreamSynchronize(stream_list[id]);
    CHECK(cudaGetLastError());


    // Lấy kết quả cuối cùng (Layer 5)
    CHECK(cudaMemcpyAsync(h_output_L5_list[id], d_output_L5_list[id],
        width_L5 * height_L5 * L5_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost, stream_list[id]));


    cudaStreamDestroy(stream_list[id]);

    // ----------------------------------------------------------------
}




float MainCompilerWithStream::runWithStream(int blockSizeVal1, int blockSizeVal3) {

    cudaStream_t* stream_list = new cudaStream_t[streamCount];

    dim3 blockSize1(blockSizeVal1, blockSizeVal1);
    dim3 gridSize1((NEW_WIDTH + blockSize1.x - 1) / blockSize1.x, (NEW_HEIGHT + blockSize1.y - 1) / blockSize1.y);

    dim3 blockSize3(blockSizeVal3, blockSizeVal3);
    dim3 gridSize3((width_L3 + blockSize3.x - 1) / blockSize3.x, (height_L3 + blockSize3.y - 1) / blockSize3.y);

    std::cout << "Start running\n";

    GpuTimer timer;
    timer.Start();

    for (int i = 0; i < 10000; i++) {
        handleRunOnDeviceWithStream(i, blockSizeVal1, blockSizeVal3,
            blockSize1, gridSize1, blockSize3, gridSize3, i % streamCount);

        if (i % 1000 == 0) {
            std::cout << "Done about " << int(i / 1000) * 10 << "%, i = " << i << "\n";
        }
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    return timer.Elapsed();
}



void MainCompilerWithStream::loadKernel() {

    int L1_kernelSize = L1_FILTERS * L1_FILTER_SIZE * L1_FILTER_SIZE;
    h_kernel_L1_list = new float* [streamCount];
    for (int i = 0; i < streamCount; i++)
        h_kernel_L1_list[i] = new float[L1_kernelSize];

    int L3_kernelSize = L3_FILTERS * L3_FILTER_SIZE * L3_FILTER_SIZE * L1_FILTERS;
    h_kernel_L3_list = new float* [streamCount];
    for (int i = 0; i < streamCount; i++)
        h_kernel_L3_list[i] = new float[L3_kernelSize];


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
                    for (int i = 0; i < streamCount; i++)
                        h_kernel_L1_list[i][kernel_pos] = temp;
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
                        for (int i = 0; i < streamCount; i++)
                            h_kernel_L3_list[i][kernel_pos] = temp;
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