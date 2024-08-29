#include "Properties.cuh"



int adjustSize(int oldSize, int kernelSize, int stride) {
    return (oldSize - kernelSize) / stride + 1;
}



void assignMemory(float* assigned, float* assigning, int width, int height, int channels) {
    for (unsigned int c = 0; c < channels; c++) {
        for (unsigned int i = 0; i < height; i++) {
            for (unsigned int j = 0; j < width; j++) {
                assigned[(c * height + i) * width + j] = assigning[(c * height + i) * width + j];
            }
        }
    }
}



float checkAccuracy(float* first, float* second, int width, int height, int channels) {
    float sum = 0.0;
    for (unsigned int c = 0; c < channels; c++) {
        for (unsigned int i = 0; i < height; i++) {
            for (unsigned int j = 0; j < width; j++) {
                sum += std::abs(first[(c * height + i) * width + j] - second[(c * height + i) * width + j]);
            }
        }
    }
    return sum;
}