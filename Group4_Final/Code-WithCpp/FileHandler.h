#ifndef FILE_HANDLER_
#define FILE_HANDLER_


#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "ImageData.h"



uint32_t reverseBytes(uint32_t bytes);
void readMnistImages(const std::string& filename, std::vector<ImageData>& images);
std::vector<ImageData> loadAndPreprocessData(const std::vector<ImageData>& images, int newWidth, int newHeight);
std::vector<uint8_t> padImage(std::vector<uint8_t> paddedImage, int oldWidth, int oldHeight, int newWidth, int newHeight);



#endif