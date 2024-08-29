#include "ConvolutionHost.h"



void normalizeImageByHost(float* image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image[i * width + j] /= 255;
        }
    }
}



void runConvolutionByHost(float* oldImage, int oldWidth, int oldHeight,
        float* newImage, int newWidth, int newHeight, 
        float* kernel, int kernelWidth, int kernelHeight, int oldChannels, int newChannels) {

    for (int i = 0; i < newHeight; i++) {
        for (int j = 0; j < newWidth; j++) {
            for (int c = 0; c < newChannels; c++) {

                float sum = 0.0;

                for (int ki = 0; ki < kernelHeight; ki++) {
                    int oi = i + ki;

                    for (int kj = 0; kj < kernelWidth; kj++) {
                        int oj = j + kj;

                        for (int kc = 0; kc < oldChannels; kc++) {
                            int oc = kc;

                            int kernelId = (c * kernelHeight * kernelWidth * oldChannels) +
                                (ki * kernelWidth * oldChannels) +
                                (kj * oldChannels) +
                                oc;

                            sum += oldImage[(oc * oldHeight + oi) * oldWidth + oj] * kernel[kernelId];
                        }
                    }
                }

                newImage[(c * newHeight + i) * newWidth + j] = sum;
            }
        }
    }
}




void runConvolutionByHostWithLargerImage(float* oldImage, int oldWidth, int oldHeight,
        float* newImage, int newWidth, int newHeight,
        float* kernel, int kernelWidth, int kernelHeight, int oldChannels, int newChannels,
        int greatInputXCount, int greatInputYCount) {

    for (int imageIdY = 0; imageIdY < greatInputYCount; imageIdY++) {
        for (int imageIdX = 0; imageIdX < greatInputXCount; imageIdX++) {

            for (int i = 0; i < newHeight; i++) {
                for (int j = 0; j < newWidth; j++) {
                    for (int c = 0; c < newChannels; c++) {

                        float sum = 0.0;
                        for (int ki = 0; ki < kernelHeight; ki++) {
                            int oi = i + imageIdY * oldHeight + ki;

                            for (int kj = 0; kj < kernelWidth; kj++) {
                                int oj = j + imageIdX * oldWidth + kj;

                                for (int kc = 0; kc < oldChannels; kc++) {
                                    int oc = kc;

                                    int kernelId = (c * kernelHeight * kernelWidth * oldChannels) +
                                        (ki * kernelWidth * oldChannels) +
                                        (kj * oldChannels) +
                                        oc;

                                    sum += oldImage[(oc * oldHeight + oi) * oldWidth + oj] * kernel[kernelId];
                                }
                            }
                        }

                        newImage[(c * newHeight + imageIdY * newHeight + i) * newWidth + imageIdX * newWidth + j] = sum;
                    }
                }
            }

        }
    }
}



void runMaxPoolingByHost(float* oldImage, int oldWidth, int oldHeight,
        float* newImage, int newWidth, int newHeight,
        int poolWidth, int poolHeight, int channels, int stride) {

    for (int i = 0; i < newHeight; i++) {
        for (int j = 0; j < newWidth; j++) {
            for (int c = 0; c < channels; c++) {

                float maxVal = -FLT_MAX;

                for (int pi = 0; pi < poolHeight; pi++) {
                    int oi = i * stride + pi;
                    if (oi >= oldHeight)
                        break;

                    for (int pj = 0; pj < poolWidth; pj++) {
                        int oj = j * stride + pj;
                        if (oj >= oldWidth)
                            break;

                        float val = oldImage[(c * oldHeight + oi) * oldWidth + oj];
                        if (val > maxVal) {
                            maxVal = val;
                        }
                    }
                }

                newImage[(c * newHeight + i) * newWidth + j] = maxVal;
            }
        }
    }
}



void ReLUByHost(float* oldImage, float* newImage, int width, int height, int channels) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int c = 0; c < channels; c++) {
                newImage[(c * height + i) * width + j] = std::max(oldImage[(c * height + i) * width + j], 0.0f);
            }
        }
    }
}

