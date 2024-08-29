#ifndef __CUDACC__  
#define __CUDACC__
#endif



#ifndef CONVOLUTION_DEVICE_
#define CONVOLUTION_DEVICE_



#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <float.h>

#include "Properties.cuh"



// Store the kernel in constant memory for fast access
__constant__ float constant_d_kernel_1[L1_FILTERS * L1_FILTER_SIZE * L1_FILTER_SIZE];
__constant__ float constant_d_kernel_3[L3_FILTERS * L3_FILTER_SIZE * L3_FILTER_SIZE * L1_FILTERS];


__global__ void basicConvolutionByDevice(
	float* input, int oldWidth, int oldHeight,
	float* output, int newWidth, int newHeight,
	float* kernel, int kernelWidth, int kernelHeight, int oldChannels, int newChannels);

__global__ void optimizedConvolutionByDevice(
	float* input, int oldWidth, int oldHeight,
	float* output, int newWidth, int newHeight,
	float* kernel, int kernelWidth, int kernelHeight, int oldChannels, int newChannels);

__global__ void optimized2ConvolutionByDevice_K1(
	float* input, int oldWidth, int oldHeight,
	float* output, int newWidth, int newHeight,
	int kernelWidth, int kernelHeight, int oldChannels, int newChannels);

__global__ void optimized2ConvolutionByDevice_K3(
	float* input, int oldWidth, int oldHeight,
	float* output, int newWidth, int newHeight,
	int kernelWidth, int kernelHeight, int oldChannels, int newChannels);

__global__ void optimizedConvolutionWithStream(
	float* input, int oldWidth, int oldHeight,
	float* output, int newWidth, int newHeight,
	float* kernel, int kernelWidth, int kernelHeight, int oldChannels);

__global__ void optimizedConvolutionWithBiggerImage(
	float* input, int oldWidth, int oldHeight,
	float* output, int newWidth, int newHeight,
	float* kernel, int kernelWidth, int kernelHeight, int oldChannels, int newChannels,
	int greatInputXCount, int greatInputYCount);

__global__ void maxPoolingByDevice(float* oldImage, int oldWidth, int oldHeight,
	float* newImage, int newWidth, int newHeight,
	int poolWidth, int poolHeight, int channels, int stride = 2);



__global__ void ReLUByDevice(float* oldImage, float* newImage, int width, int height, int channels);



void assignConstantKernel1(float* h_kernel_L1);
void assignConstantKernel3(float* h_kernel_L3);

//void assignConstantKernel3(float* h_kernel_L1) {
//	cudaMemcpyToSymbol(constant_d_kernel_1, h_kernel_L1, L1_FILTERS * L1_FILTER_SIZE * L1_FILTER_SIZE * sizeof(float));
//}




__global__ void basicConvolutionByDevice(
        float* input, int oldWidth, int oldHeight,
        float* output, int newWidth, int newHeight,
        float* kernel, int kernelWidth, int kernelHeight, int oldChannels, int newChannels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < newWidth && y < newHeight) {

        for (int z = 0; z < newChannels; z++) {

            float sum = 0;

            for (int ki = 0; ki < kernelHeight; ki++) {
                int oi = y + ki;

                for (int kj = 0; kj < kernelWidth; kj++) {
                    int oj = x + kj;

                    for (int kc = 0; kc < oldChannels; kc++) {
                        int oc = kc;        // Old channels

                        int kernelId = (z * kernelHeight * kernelWidth * oldChannels) +
                            (ki * kernelWidth * oldChannels) +
                            (kj * oldChannels) +
                            oc;

                        sum += input[(oc * oldHeight + oi) * oldWidth + oj] * kernel[kernelId];
                    }
                }
            }

            output[(z * newHeight + y) * newWidth + x] = sum;
        }
    }
}




__global__ void optimizedConvolutionByDevice(
        float* input, int oldWidth, int oldHeight,
        float* output, int newWidth, int newHeight,
        float* kernel, int kernelWidth, int kernelHeight, int oldChannels, int newChannels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int conv_size_x = blockDim.x + kernelWidth - 1;
    const int conv_size_y = blockDim.y + kernelHeight - 1;

    extern __shared__ float small_input[];

    

    // Assign to SMEM variable
    for (int z = 0; z < oldChannels; z++) {
        for (int group_y = 0; group_y < (conv_size_y - 1) / blockDim.y + 1; group_y++) {
            int ci = group_y * blockDim.y + threadIdx.y;
            if (ci >= conv_size_y)
                break;

            int as_y = y + group_y * blockDim.y;
            for (int group_x = 0; group_x < (conv_size_x - 1) / blockDim.x + 1; group_x++) {
                int cj = group_x * blockDim.x + threadIdx.x;
                if (cj >= conv_size_x)
                    break;

                int as_x = x + group_x * blockDim.x;
                small_input[(z * conv_size_y + ci) * conv_size_x + cj] = input[(z * oldHeight + as_y) * oldWidth + as_x];
            }
        }
    }
    __syncthreads();



    if ((x < newWidth) && (y < newHeight)) {

        for (int z = 0; z < newChannels; z++) {

            float sum = 0;

            for (int ki = 0; ki < kernelHeight; ki++) {
                int oi = threadIdx.y + ki;

                for (int kj = 0; kj < kernelWidth; kj++) {
                    int oj = threadIdx.x + kj;

                    for (int kc = 0; kc < oldChannels; kc++) {
                        int oc = kc;

                        int kernelId = (z * kernelHeight * kernelWidth * oldChannels) +
                            (ki * kernelWidth * oldChannels) +
                            (kj * oldChannels) +
                            oc;

                        sum += small_input[(oc * conv_size_y + oi) * conv_size_x + oj] * kernel[kernelId];
                    }
                }
            }

            output[(z * newHeight + y) * newWidth + x] = sum;
        }
        __syncthreads();
    }
}



__global__ void optimized2ConvolutionByDevice_K1(
        float* input, int oldWidth, int oldHeight,
        float* output, int newWidth, int newHeight,
        int kernelWidth, int kernelHeight, int oldChannels, int newChannels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int conv_size_x = blockDim.x + kernelWidth - 1;
    const int conv_size_y = blockDim.y + kernelHeight - 1;

    extern __shared__ float small_input[];



    // Assign to SMEM variable
    for (int z = 0; z < oldChannels; z++) {
        for (int group_y = 0; group_y < (conv_size_y - 1) / blockDim.y + 1; group_y++) {
            int ci = group_y * blockDim.y + threadIdx.y;
            if (ci >= conv_size_y)
                break;

            int as_y = y + group_y * blockDim.y;
            for (int group_x = 0; group_x < (conv_size_x - 1) / blockDim.x + 1; group_x++) {
                int cj = group_x * blockDim.x + threadIdx.x;
                if (cj >= conv_size_x)
                    break;

                int as_x = x + group_x * blockDim.x;
                small_input[(z * conv_size_y + ci) * conv_size_x + cj] = input[(z * oldHeight + as_y) * oldWidth + as_x];
            }
        }
    }
    __syncthreads();



    if ((x < newWidth) && (y < newHeight)) {

        for (int z = 0; z < newChannels; z++) {

            float sum = 0;

            for (int ki = 0; ki < kernelHeight; ki++) {
                int oi = threadIdx.y + ki;

                for (int kj = 0; kj < kernelWidth; kj++) {
                    int oj = threadIdx.x + kj;

                    for (int kc = 0; kc < oldChannels; kc++) {
                        int oc = kc;

                        int kernelId = (z * kernelHeight * kernelWidth * oldChannels) +
                            (ki * kernelWidth * oldChannels) +
                            (kj * oldChannels) +
                            oc;

                        sum += small_input[(oc * conv_size_y + oi) * conv_size_x + oj] * constant_d_kernel_1[kernelId];
                    }
                }
            }

            output[(z * newHeight + y) * newWidth + x] = sum;
        }
        __syncthreads();
    }
}



__global__ void optimized2ConvolutionByDevice_K3(
    float* input, int oldWidth, int oldHeight,
    float* output, int newWidth, int newHeight,
    int kernelWidth, int kernelHeight, int oldChannels, int newChannels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int conv_size_x = blockDim.x + kernelWidth - 1;
    const int conv_size_y = blockDim.y + kernelHeight - 1;

    extern __shared__ float small_input[];



    // Assign to SMEM variable
    for (int z = 0; z < oldChannels; z++) {
        for (int group_y = 0; group_y < (conv_size_y - 1) / blockDim.y + 1; group_y++) {
            int ci = group_y * blockDim.y + threadIdx.y;
            if (ci >= conv_size_y)
                break;

            int as_y = y + group_y * blockDim.y;
            for (int group_x = 0; group_x < (conv_size_x - 1) / blockDim.x + 1; group_x++) {
                int cj = group_x * blockDim.x + threadIdx.x;
                if (cj >= conv_size_x)
                    break;

                int as_x = x + group_x * blockDim.x;
                small_input[(z * conv_size_y + ci) * conv_size_x + cj] = input[(z * oldHeight + as_y) * oldWidth + as_x];
            }
        }
    }
    __syncthreads();



    if ((x < newWidth) && (y < newHeight)) {

        for (int z = 0; z < newChannels; z++) {

            float sum = 0;

            for (int ki = 0; ki < kernelHeight; ki++) {
                int oi = threadIdx.y + ki;

                for (int kj = 0; kj < kernelWidth; kj++) {
                    int oj = threadIdx.x + kj;

                    for (int kc = 0; kc < oldChannels; kc++) {
                        int oc = kc;

                        int kernelId = (z * kernelHeight * kernelWidth * oldChannels) +
                            (ki * kernelWidth * oldChannels) +
                            (kj * oldChannels) +
                            oc;

                        sum += small_input[(oc * conv_size_y + oi) * conv_size_x + oj] * constant_d_kernel_3[kernelId];
                    }
                }
            }

            output[(z * newHeight + y) * newWidth + x] = sum;
        }
        __syncthreads();
    }
}



__global__ void optimizedConvolutionWithStream(
    float* input, int oldWidth, int oldHeight,
    float* output, int newWidth, int newHeight,
    float* kernel, int kernelWidth, int kernelHeight, int oldChannels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int conv_size_x = blockDim.x + kernelWidth - 1;
    const int conv_size_y = blockDim.y + kernelHeight - 1;

    extern __shared__ float small_input[];


    // Assign to SMEM variable
    for (int z = 0; z < oldChannels; z++) {
        for (int group_y = 0; group_y < (conv_size_y - 1) / blockDim.y + 1; group_y++) {
            int ci = group_y * blockDim.y + threadIdx.y;
            if (ci >= conv_size_y)
                break;

            int as_y = y + group_y * blockDim.y;
            for (int group_x = 0; group_x < (conv_size_x - 1) / blockDim.x + 1; group_x++) {
                int cj = group_x * blockDim.x + threadIdx.x;
                if (cj >= conv_size_x)
                    break;

                int as_x = x + group_x * blockDim.x;
                small_input[(z * conv_size_y + ci) * conv_size_x + cj] = input[(z * oldHeight + as_y) * oldWidth + as_x];
            }
        }
    }
    __syncthreads();


    if ((x < newWidth) && (y < newHeight)) {

        float sum = 0;

        for (int ki = 0; ki < kernelHeight; ki++) {
            int oi = threadIdx.y + ki;

            for (int kj = 0; kj < kernelWidth; kj++) {
                int oj = threadIdx.x + kj;

                for (int kc = 0; kc < oldChannels; kc++) {
                    int oc = kc;

                    int kernelId = (ki * kernelWidth * oldChannels) +
                        (kj * oldChannels) +
                        oc;

                    sum += small_input[(oc * conv_size_y + oi) * conv_size_x + oj] * kernel[kernelId];
                }
            }
        }

        output[y * newWidth + x] = sum;
    }
}



__global__ void optimizedConvolutionWithBiggerImage(
        float* input, int oldWidth, int oldHeight,
        float* output, int newWidth, int newHeight,
        float* kernel, int kernelWidth, int kernelHeight, int oldChannels, int newChannels,
        int greatInputXCount, int greatInputYCount) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int conv_size_x = blockDim.x + kernelWidth - 1;
    const int conv_size_y = blockDim.y + kernelHeight - 1;

    extern __shared__ float small_input[];



    // Assign to SMEM variable
    // Nothing to change compared to 'optimizedConvolutionByDevice'
    // Except: the location of the cell of the input
    for (int z = 0; z < oldChannels; z++) {
        for (int group_y = 0; group_y < (conv_size_y - 1) / blockDim.y + 1; group_y++) {
            int ci = group_y * blockDim.y + threadIdx.y;
            if (ci >= conv_size_y)
                break;

            int as_y = y + group_y * blockDim.y;
            for (int group_x = 0; group_x < (conv_size_x - 1) / blockDim.x + 1; group_x++) {
                int cj = group_x * blockDim.x + threadIdx.x;
                if (cj >= conv_size_x)
                    break;

                int as_x = x + group_x * blockDim.x;
                int inputId = (z * oldHeight * greatInputYCount + as_y) * oldWidth * greatInputXCount + as_x;
                small_input[(z * conv_size_y + ci) * conv_size_x + cj] = input[inputId];
            }
        }
    }
    __syncthreads();



    // Unnecessary cell with be ignored
    // Eg: Image with size of (14 x 14), block size of (8 x 8), and the kernel is (5 x 5)
    //     At block id = 0: Cell [0, 0] to [0, 7] will be used
    //     At block id = 1: Cell [0, 8], [0, 9], [0, 14], [0, 15] will be used. The rest is ignored
    // 
    // Formula: x % old width < new height
    // Test:    x % 14 < 10: Value 0 - 7 valid; 8, 9, 14, 15 valid
    //
    if ((x < newWidth * greatInputXCount) && (y < newHeight * greatInputYCount) &&
            (x % oldWidth < newWidth) && (y % oldHeight < newHeight)) {

        for (int z = 0; z < newChannels; z++) {

            float sum = 0;

            for (int ki = 0; ki < kernelHeight; ki++) {
                int oi = threadIdx.y + ki;

                for (int kj = 0; kj < kernelWidth; kj++) {
                    int oj = threadIdx.x + kj;

                    for (int kc = 0; kc < oldChannels; kc++) {
                        int oc = kc;

                        int kernelId = (z * kernelHeight * kernelWidth * oldChannels) +
                            (ki * kernelWidth * oldChannels) +
                            (kj * oldChannels) +
                            oc;

                        sum += small_input[(oc * conv_size_y + oi) * conv_size_x + oj] * kernel[kernelId];
                    }
                }
            }

            int outputId = (z * newHeight * greatInputYCount + y - y / newHeight * (oldHeight - newHeight))
                * newWidth * greatInputXCount + x - x / newWidth * (oldWidth - newWidth);
            output[outputId] = sum;

            //if (blockIdx.x == 0 && blockIdx.y == 0 && z == 0) {
            //if (outputId == 2800) {
            //    printf("At [%d, %d], or [%d] ->\t{%.0f}\n", y, x, (z * newHeight * greatInputYCount + y) * newWidth * greatInputXCount + x, sum);
            //}
        }
        __syncthreads();
    }

    //if (newChannels == 16 && threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1 && blockIdx.y == 0) {
    //    printf("\n\n\n");
    //    for (int i = 0; i < 14; i++) {
    //        for (int j = 0; j < 14; j++) {
    //            printf("%.3f, ", output[(i + blockIdx.x * blockDim.x) * newWidth * greatInputXCount + j]);
    //        }
    //        printf("\n");
    //    }

    //    //printf("\n\n\n");
    //    //for (int i = 0; i < conv_size_y; i++) {
    //    //    for (int j = 0; j < conv_size_x; j++) {
    //    //        printf("At [%d, %d], or [%d] ->\t{%.0f}\n", i, j, i* newWidth* greatInputXCount + j, output[i * newWidth * greatInputXCount + j]);
    //    //    }
    //    //    printf("\n");
    //    //}
    //}
}



__global__ void maxPoolingByDevice(float* oldImage, int oldWidth, int oldHeight,
        float* newImage, int newWidth, int newHeight,
        int poolWidth, int poolHeight, int channels, int stride) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int z = 0; z < channels; z++) {

        float maxVal = -FLT_MAX;
        bool assigning = false;

        for (int pi = 0; pi < poolHeight; pi++) {
            int oi = y * stride + pi;
            if (oi >= oldHeight)
                break;

            for (int pj = 0; pj < poolWidth; pj++) {
                int oj = x * stride + pj;
                if (oj >= oldWidth)
                    break;

                float val = oldImage[(z * oldHeight + oi) * oldWidth + oj];
                if (val > maxVal) {
                    maxVal = val;
                    assigning = true;
                }
            }
        }

        if (assigning)
            newImage[(z * newHeight + y) * newWidth + x] = maxVal;
    }
}



__global__ void ReLUByDevice(float* oldImage, float* newImage, int width, int height, int channels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int z = 0; z < channels; z++) {
            float maxVal = max(oldImage[(z * height + y) * width + x], 0.0f);
            newImage[(z * height + y) * width + x] = maxVal;
        }
    }
}




void assignConstantKernel1(float* h_kernel_L1) {
    cudaMemcpyToSymbol(constant_d_kernel_1, h_kernel_L1, L1_FILTERS * L1_FILTER_SIZE * L1_FILTER_SIZE * sizeof(float));
}

void assignConstantKernel3(float* h_kernel_L3) {
    cudaMemcpyToSymbol(constant_d_kernel_3, h_kernel_L3, L3_FILTERS * L3_FILTER_SIZE * L3_FILTER_SIZE * L1_FILTERS * sizeof(float));
}





#endif