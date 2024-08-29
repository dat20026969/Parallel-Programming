#ifndef CONVOLUTION_HOST_
#define CONVOLUTION_HOST_



#include <vector>
#include <algorithm>

#include "ImageData.h"
#include "FileHandler.h"



void normalizeImageByHost(float* image, int width, int height);
void runConvolutionByHost(float* oldImage, int oldWidth, int oldHeight,
    float* newImage, int newWidth, int newHeight,
    float* kernel, int kernelWidth, int kernelHeight, int oldChannels, int newChannels);
void runConvolutionByHostWithLargerImage(float* oldImage, int oldWidth, int oldHeight,
    float* newImage, int newWidth, int newHeight,
    float* kernel, int kernelWidth, int kernelHeight, int oldChannels, int newChannels,
    int greatInputXCount, int greatInputYCount);
void runMaxPoolingByHost(float* oldImage, int oldWidth, int oldHeight,
    float* newImage, int newWidth, int newHeight,
    int poolWidth, int poolHeight, int channels, int stride = 2);

void ReLUByHost(float* oldImage, float* newImage, int width, int height, int channels);



#endif