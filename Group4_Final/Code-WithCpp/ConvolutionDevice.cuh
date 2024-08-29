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




#endif