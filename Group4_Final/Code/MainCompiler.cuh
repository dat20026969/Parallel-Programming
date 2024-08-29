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



void randomElementKernel(float* arr, int n) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(1.0 / n * 5, 0.03);
    for (int i = 0; i < n; i++) {
        // arr[i] = (float)(rand()) / (float)(RAND_MAX);
        arr[i] = distribution(generator);
    }
}



MainCompiler::MainCompiler() {
    srand(time(nullptr));
}



MainCompiler::~MainCompiler() {

    if (d_input == NULL)
        return;

    delete[] h_input;
    delete[] h_output_L2;
    delete[] h_output_L3;
    delete[] h_output_L4;
    delete[] h_output_L5;

    delete[] h_kernel_L1;
    delete[] h_kernel_L3;

    cudaFree(d_input);
    cudaFree(d_output_L2);
    cudaFree(d_output_L3);
    cudaFree(d_output_L4);
    cudaFree(d_output_L5);

    cudaFree(d_kernel_L1);
    cudaFree(d_kernel_L3);
}




void MainCompiler::readFile(std::string trainImagesFilename, std::string testImagesFilename) {
    readMnistImages(trainImagesFilename, trainData);
    readMnistImages(testImagesFilename, testData);

    trainData = loadAndPreprocessData(trainData, NEW_WIDTH, NEW_HEIGHT);
    testData = loadAndPreprocessData(testData, NEW_WIDTH, NEW_HEIGHT);

    std::cout << testData[0].image.size();
}



void MainCompiler::assignHostMemory() {

    width_L2  = adjustSize(NEW_WIDTH,  L1_FILTER_SIZE, 1);
    height_L2 = adjustSize(NEW_HEIGHT, L1_FILTER_SIZE, 1);
    width_L3  = adjustSize(width_L2,  L2_POOL_SIZE, 2);
    height_L3 = adjustSize(height_L2, L2_POOL_SIZE, 2);
    width_L4  = adjustSize(width_L3,  L3_FILTER_SIZE, 1);
    height_L4 = adjustSize(height_L3, L3_FILTER_SIZE, 1);
    width_L5 =  adjustSize(width_L4,  L4_POOL_SIZE, 2);
    height_L5 = adjustSize(height_L4, L4_POOL_SIZE, 2);

    std::cout << "\nImage size at layer 1: (" << NEW_WIDTH << ", " << NEW_HEIGHT << ", " << L1_CHANNELS << ")\n";
    std::cout << "Image size at layer 2: (" << width_L2 << ", " << height_L2 << ", " << L2_CHANNELS << ")\n";
    std::cout << "Image size at layer 3: (" << width_L3 << ", " << height_L3 << ", " << L3_CHANNELS << ")\n";
    std::cout << "Image size at layer 4: (" << width_L4 << ", " << height_L4 << ", " << L4_CHANNELS << ")\n";
    std::cout << "Image size at layer 5: (" << width_L5 << ", " << height_L5 << ", " << L5_CHANNELS << ")\n\n";

    h_input = new float[NEW_WIDTH * NEW_HEIGHT];
    h_output_L2 = new float[L2_CHANNELS * width_L2 * height_L2];
    h_output_L3 = new float[L3_CHANNELS * width_L3 * height_L3];
    h_output_L4 = new float[L4_CHANNELS * width_L4 * height_L4];
    h_output_L5 = new float[L5_CHANNELS * width_L5 * height_L5];
}



void MainCompiler::assignDeviceMemory() {

    CHECK(cudaMalloc(&d_input, NEW_WIDTH * NEW_HEIGHT * sizeof(float)));


    // Lớp 1: Tích chập
    int L1_kernelSize = L1_FILTERS * L1_FILTER_SIZE * L1_FILTER_SIZE;
    CHECK(cudaMalloc(&d_output_L2, L2_CHANNELS * width_L2 * height_L2 * sizeof(float)));
    CHECK(cudaMalloc(&d_kernel_L1, L1_kernelSize * sizeof(float)));
    CHECK(cudaMemcpy(d_kernel_L1, h_kernel_L1, L1_kernelSize * sizeof(float), cudaMemcpyHostToDevice));
    


    // Lớp 2: Max Pooling
    CHECK(cudaMalloc(&d_output_L3, width_L3 * height_L3 * L3_CHANNELS * sizeof(float)));


    // Lớp 3: Tích chập thứ hai
    int L3_kernelSize = L3_FILTERS * L3_FILTER_SIZE * L3_FILTER_SIZE * L1_FILTERS;  // Nhân với L1_FILTERS vì lớp này nhận đầu vào từ nhiều bộ lọc
    CHECK(cudaMalloc(&d_output_L4, L4_CHANNELS * width_L4 * height_L4 * sizeof(float)));
    CHECK(cudaMalloc(&d_kernel_L3, L3_kernelSize * sizeof(float)));
    CHECK(cudaMemcpy(d_kernel_L3, h_kernel_L3, L3_kernelSize * sizeof(float), cudaMemcpyHostToDevice));


    // Lớp 4: Max Pooling thứ hai
    CHECK(cudaMalloc(&d_output_L5, width_L5 * height_L5 * L5_CHANNELS * sizeof(float)));
}



void MainCompiler::handleRunOnHost(int& image_id) {

    // Before
    for (unsigned int i = 0; i < NEW_HEIGHT; i++) {
        for (unsigned int j = 0; j < NEW_WIDTH; j++) {
            h_input[i * NEW_WIDTH + j] = testData[image_id].image[i * NEW_WIDTH + j];
        }
    }

    // Tiền xử lý
    // resizeImageByHost(h_input, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_NEW_WIDTH, IMAGE_NEW_HEIGHT);
    // normalizeImageByHost(h_input, IMAGE_NEW_WIDTH, IMAGE_NEW_HEIGHT);

    // Lớp 1
    runConvolutionByHost(h_input, NEW_WIDTH, NEW_HEIGHT,
        h_output_L2, width_L2, height_L2,
        h_kernel_L1, L1_FILTER_SIZE, L1_FILTER_SIZE, L1_CHANNELS, L2_CHANNELS);

    ReLUByHost(h_output_L2, h_output_L2, width_L2, height_L2, L2_CHANNELS);


    // Lớp 2
    runMaxPoolingByHost(h_output_L2, width_L2, height_L2,
        h_output_L3, width_L3, height_L3, L2_POOL_SIZE, L2_POOL_SIZE, L3_CHANNELS);


    // Lớp 3
    runConvolutionByHost(h_output_L3, width_L3, height_L3,
        h_output_L4, width_L4, height_L4,
        h_kernel_L3, L3_FILTER_SIZE, L3_FILTER_SIZE, L3_CHANNELS, L4_CHANNELS);

    ReLUByHost(h_output_L4, h_output_L4, width_L4, height_L4, L4_CHANNELS);


    // Lớp 4
    runMaxPoolingByHost(h_output_L4, width_L4, height_L4,
        h_output_L5, width_L5, height_L5, L4_POOL_SIZE, L4_POOL_SIZE, L5_CHANNELS);
}



void MainCompiler::handleRunOnDevice(int& image_id, int& compiling_type, int& blockSizeVal1, int blockSizeVal3,
        dim3& blockSize1, dim3& gridSize1, dim3& blockSize3, dim3& gridSize3) {

    // Before
    for (unsigned int i = 0; i < NEW_HEIGHT; i++) {
        for (unsigned int j = 0; j < NEW_WIDTH; j++) {
            h_input[i * NEW_WIDTH + j] = testData[image_id].image[i * NEW_WIDTH + j];
        }
    }

    // Truyền input vào
    CHECK(cudaMemcpy(d_input, h_input, NEW_WIDTH * NEW_HEIGHT * sizeof(float), cudaMemcpyHostToDevice));


    // -----------------------------------------------------------------


    // ----------------------------- Lớp 1 ----------------------------- 
    size_t sharedSize1;
    switch (compiling_type) {
    case 1:
        basicConvolutionByDevice <<< gridSize1, blockSize1 >>> (
            d_input, NEW_WIDTH, NEW_HEIGHT,
            d_output_L2, width_L2, height_L2,
            d_kernel_L1, L1_FILTER_SIZE, L1_FILTER_SIZE, L1_CHANNELS, L2_CHANNELS
            );
        break;

    case 2:
        sharedSize1 = (blockSizeVal1 + L1_FILTER_SIZE - 1) *
            (blockSizeVal1 + L1_FILTER_SIZE - 1) * L2_CHANNELS * sizeof(float);

        optimizedConvolutionByDevice <<< gridSize1, blockSize1, sharedSize1 >>> (
            d_input, NEW_WIDTH, NEW_HEIGHT,
            d_output_L2, width_L2, height_L2,
            d_kernel_L1, L1_FILTER_SIZE, L1_FILTER_SIZE, L1_CHANNELS, L2_CHANNELS
            );
        break;

    case 3:
        sharedSize1 = (blockSizeVal1 + L1_FILTER_SIZE - 1) *
            (blockSizeVal1 + L1_FILTER_SIZE - 1) * L2_CHANNELS * sizeof(float);

        optimized2ConvolutionByDevice_K1 <<< gridSize1, blockSize1, sharedSize1 >>> (
            d_input, NEW_WIDTH, NEW_HEIGHT,
            d_output_L2, width_L2, height_L2,
            L1_FILTER_SIZE, L1_FILTER_SIZE, L1_CHANNELS, L2_CHANNELS
            );
        break;
    }
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // ReLU
    ReLUByDevice <<< gridSize1, blockSize1 >>> (d_output_L2, d_output_L2, width_L2, height_L2, L2_CHANNELS);

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());


    // -----------------------------  Lớp 2 ----------------------------- 
    maxPoolingByDevice <<< gridSize1, blockSize1 >>> (
        d_output_L2, width_L2, height_L2,
        d_output_L3, width_L3, height_L3,
        L2_POOL_SIZE, L2_POOL_SIZE, L3_CHANNELS, 2);

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());


    // ----------------------------- Lớp 3 ----------------------------- 
    size_t sharedSize3;
    switch (compiling_type) {
    case 1:
        basicConvolutionByDevice <<< gridSize3, blockSize3 >>> (
            d_output_L3, width_L3, height_L3,
            d_output_L4, width_L4, height_L4,
            d_kernel_L3, L3_FILTER_SIZE, L3_FILTER_SIZE, L3_CHANNELS, L4_CHANNELS
            );
        break;

    case 2:
        sharedSize3 = (blockSizeVal3 + L3_FILTER_SIZE - 1) *
            (blockSizeVal3 + L3_FILTER_SIZE - 1) * L4_CHANNELS * sizeof(float);

        optimizedConvolutionByDevice <<< gridSize3, blockSize3, sharedSize3 >>> (
            d_output_L3, width_L3, height_L3,
            d_output_L4, width_L4, height_L4,
            d_kernel_L3, L3_FILTER_SIZE, L3_FILTER_SIZE, L3_CHANNELS, L4_CHANNELS
            );
        break;

    case 3:
        sharedSize3 = (blockSizeVal3 + L3_FILTER_SIZE - 1) *
            (blockSizeVal3 + L3_FILTER_SIZE - 1) * L4_CHANNELS * sizeof(float);

        optimized2ConvolutionByDevice_K3 <<< gridSize3, blockSize3, sharedSize3 >>> (
            d_output_L3, width_L3, height_L3,
            d_output_L4, width_L4, height_L4,
            L3_FILTER_SIZE, L3_FILTER_SIZE, L3_CHANNELS, L4_CHANNELS
            );
        break;
    }
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    ReLUByDevice <<< gridSize3, blockSize3 >>> (d_output_L4, d_output_L4, width_L4, height_L4, L4_CHANNELS);

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());


    // ----------------------------- Lớp 4 ----------------------------- 
    maxPoolingByDevice <<< gridSize3, blockSize3 >>> (
        d_output_L4, width_L4, height_L4, d_output_L5, width_L5, height_L5,
        L4_POOL_SIZE, L4_POOL_SIZE, L4_CHANNELS, 2);

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());


    // Lấy kết quả cuối cùng (Layer 5)
    CHECK(cudaMemcpy(h_output_L5, d_output_L5, width_L5 * height_L5 * L5_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost));


    // ----------------------------------------------------------------
}



float MainCompiler::runOnHost(int image_id) {

    this->image_id = image_id;
    compiling_type = 0;

    GpuTimer timer;
    timer.Start();

    // image_id == -1: Run all
    if (image_id == -1) {
        for (int curr_image_id = 0; curr_image_id < 10000; curr_image_id++) {
            handleRunOnHost(curr_image_id);

            if (curr_image_id % 1000 == 0) {
                std::cout << "Done about " << int(curr_image_id / 1000) * 10 << "%, i = " << curr_image_id << "\n";
            }
        }
    }
    else {
        handleRunOnHost(image_id);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    return timer.Elapsed();
}



float MainCompiler::runOnDevice(int image_id, int compiling_type, int blockSizeVal1, int blockSizeVal3) {

    // Gán giá trị
    this->image_id = image_id;
    this->compiling_type = compiling_type;

    if (compiling_type == 3) {
        assignConstantKernel3(h_kernel_L3);
        assignConstantKernel1(h_kernel_L1);
    }

    // -----------------------------------

    // Tính thời gian cho cả việc gán biến input vào device và biến output vào host
    dim3 blockSize1(blockSizeVal1, blockSizeVal1);
    dim3 gridSize1((NEW_WIDTH + blockSize1.x - 1) / blockSize1.x, (NEW_HEIGHT + blockSize1.y - 1) / blockSize1.y);

    dim3 blockSize3(blockSizeVal3, blockSizeVal3);
    dim3 gridSize3((width_L3 + blockSize3.x - 1) / blockSize3.x, (height_L3 + blockSize3.y - 1) / blockSize3.y);

    GpuTimer timer;
    timer.Start();
    
    // image_id == -1: Run all
    if (image_id == -1) {
        for (int curr_image_id = 0; curr_image_id < 10000; curr_image_id++) {

            handleRunOnDevice(curr_image_id, compiling_type, blockSizeVal1, blockSizeVal3,
                blockSize1, gridSize1, blockSize3, gridSize3);

            if (curr_image_id % 1000 == 0) {
                std::cout << "Done about " << int(curr_image_id / 1000) * 10 << "%, i = " << curr_image_id << "\n";
            }
        }
    }
    else {
        handleRunOnDevice(image_id, compiling_type, blockSizeVal1, blockSizeVal3,
            blockSize1, gridSize1, blockSize3, gridSize3);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());


    // Lấy các giá trị còn lại
    CHECK(cudaMemcpy(h_output_L2, d_output_L2, width_L2 * height_L2 * L2_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_output_L3, d_output_L3, width_L3 * height_L3 * L3_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_output_L4, d_output_L4, width_L4 * height_L4 * L4_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost));

    //for (int c = 0; c < L5_CHANNELS; c++) {
    //    std::cout << "Channel " << c << "\n";
    //    for (int i = 0; i < height_L5; i++) {
    //        for (int j = 0; j < width_L5; j++) {
    //            printf("%.5f, ", h_output_L5[(c * height_L5 + i) * width_L5 + j]);
    //        }
    //        std::cout << "\n";
    //    }
    //    std::cout << "\n";
    //}

    return timer.Elapsed();
}



void MainCompiler::runAll(int image_id, int blockSize1, int blockSize3) {

    size_t result_size = L5_CHANNELS * width_L5 * height_L5;

    float* host_result = new float[result_size];
    float* device_result_1 = new float[result_size];
    float* device_result_2 = new float[result_size];
    float* device_result_3 = new float[result_size];
    
    // Host
    float host_time = runOnHost(image_id);
    assignMemory(host_result, h_output_L5, width_L5, height_L5, L5_CHANNELS);


    // Device 1
    float device_1_time = runOnDevice(image_id, 1, blockSize1, blockSize3);
    assignMemory(device_result_1, h_output_L5, width_L5, height_L5, L5_CHANNELS);

    // Device 2
    float device_2_time = runOnDevice(image_id, 2, blockSize1, blockSize3);
    assignMemory(device_result_2, h_output_L5, width_L5, height_L5, L5_CHANNELS);

    // Device 3
    float device_3_time = runOnDevice(image_id, 3, blockSize1, blockSize3);
    assignMemory(device_result_3, h_output_L5, width_L5, height_L5, L5_CHANNELS);

    // Check accuracy
    std::cout << "Check accuracy of the results:\n\n";
    std::cout << " -  Time spend of the host: " << host_time << "\n";
    std::cout << " -  The error of the kernel 1: "
        << checkAccuracy(host_result, device_result_1, width_L5, height_L5, L5_CHANNELS)
        << "\t. Time spend: " << device_1_time << "\n";
    std::cout << " -  The error of the kernel 2: "
        << checkAccuracy(host_result, device_result_1, width_L5, height_L5, L5_CHANNELS)
        << "\t. Time spend: " << device_2_time << "\n";
    std::cout << " -  The error of the kernel 3: "
        << checkAccuracy(host_result, device_result_1, width_L5, height_L5, L5_CHANNELS)
        << "\t. Time spend: " << device_3_time << "\n";


    delete[] host_result;
    delete[] device_result_1;
    delete[] device_result_2;
    delete[] device_result_3;
}




void MainCompiler::saveRawText() {

    const int width_list[] = { NEW_WIDTH, width_L2, width_L3, width_L4, width_L5 };
    const int height_list[] = { NEW_HEIGHT, height_L2, height_L3, height_L4, height_L5 };
    const int channel_list[] = { L1_CHANNELS, L2_CHANNELS, L3_CHANNELS, L4_CHANNELS, L5_CHANNELS };
    const float* p_layer_list[] = { h_input, h_output_L2, h_output_L3, h_output_L4, h_output_L5 };


    for (int layer = 1; layer <= 5; layer++) {

        for (int channel = 0; channel < channel_list[layer - 1]; channel++) {

            std::string file_name = "Images/Id"
                + std::to_string(image_id) + "_Type"
                + std::to_string(compiling_type) + "_L"
                + std::to_string(layer) + "_C"
                + std::to_string(channel) + ".csv";

            std::cout << " - Saving " << file_name << "\n";
            std::ofstream file(file_name);

            if (file.is_open()) {

                int height = height_list[layer - 1];
                int width = width_list[layer - 1];

                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        file << p_layer_list[layer - 1][(channel * height + i) * width + j] << ",";
                    }
                    file << "\n";
                }
            }
            else {
                std::cerr << "Unable to write such a file\n";
            }

            file.close();
        }
    }
}



void MainCompiler::saveKernel() {

    // Kernel 1
    std::string file_name = "Images/Kernel1.txt";
    std::ofstream k1_file(file_name);
    if (k1_file.is_open()) {
        // old depth, height, weight, new depth
        for (int nc = 0; nc < L2_CHANNELS; nc++) {
            for (int i = 0; i < L1_FILTER_SIZE; i++) {
                for (int j = 0; j < L1_FILTER_SIZE; j++) {
                    unsigned int kernel_pos = nc * L1_FILTER_SIZE * L1_FILTER_SIZE + j * L1_FILTER_SIZE + i;
                    k1_file << h_kernel_L1[kernel_pos] << " ";
                }
                k1_file << "\n";
            }
            k1_file << "\n";
        }
    }
    else {
        std::cerr << "Unable to write such a file " << file_name << "\n";
    }
    k1_file.close();


    file_name = "Images/Kernel3.txt";
    std::ofstream k3_file(file_name);
    if (k3_file.is_open()) {
        // old depth, height, weight, new depth
        for (int oc = 0; oc < L3_CHANNELS; oc++) {
            for (int nc = 0; nc < L4_CHANNELS; nc++) {
                for (int i = 0; i < L1_FILTER_SIZE; i++) {
                    for (int j = 0; j < L1_FILTER_SIZE; j++) {
                        unsigned int kernel_pos = (nc * L1_FILTER_SIZE * L1_FILTER_SIZE * L3_CHANNELS) +
                            (j * L1_FILTER_SIZE * L3_CHANNELS) +
                            (i * L3_CHANNELS) + oc;
                        k3_file << h_kernel_L3[kernel_pos] << " ";
                    }
                    k3_file << "\n";
                }
                k3_file << "\n";
            }
            k3_file << "\n";
        }
    }
    else {
        std::cerr << "Unable to write such a file " << file_name << "\n";
    }
    k3_file.close();
}



void MainCompiler::loadKernel() {

    int L1_kernelSize = L1_FILTERS * L1_FILTER_SIZE * L1_FILTER_SIZE;
    h_kernel_L1 = new float[L1_kernelSize];

    int L3_kernelSize = L3_FILTERS * L3_FILTER_SIZE * L3_FILTER_SIZE * L1_FILTERS;
    h_kernel_L3 = new float[L3_kernelSize];


    // Kernel 1
    std::string file_name = "Images/Kernel1.txt";
    std::ifstream k1_file(file_name);
    if (k1_file.is_open()) {
        for (int nc = 0; nc < L2_CHANNELS; nc++) {
            for (int i = 0; i < L1_FILTER_SIZE; i++) {
                for (int j = 0; j < L1_FILTER_SIZE; j++) {
                    unsigned int kernel_pos = nc * L1_FILTER_SIZE * L1_FILTER_SIZE + j * L1_FILTER_SIZE + i;
                    k1_file >> h_kernel_L1[kernel_pos];
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
                        k3_file >> h_kernel_L3[kernel_pos];
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



void MainCompiler::generateKernel() {

    int L1_kernelSize = L1_FILTERS * L1_FILTER_SIZE * L1_FILTER_SIZE;
    h_kernel_L1 = new float[L1_kernelSize];
    randomElementKernel(h_kernel_L1, L1_kernelSize);

    int L3_kernelSize = L3_FILTERS * L3_FILTER_SIZE * L3_FILTER_SIZE * L1_FILTERS;  // Nhân với L1_FILTERS vì lớp này nhận đầu vào từ nhiều bộ lọc
    h_kernel_L3 = new float[L3_kernelSize];
    randomElementKernel(h_kernel_L3, L3_kernelSize);
}




#endif