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



uint32_t reverseBytes(uint32_t bytes) {
    return (bytes >> 24) |
        ((bytes << 8) & 0x00FF0000) |
        ((bytes >> 8) & 0x0000FF00) |
        (bytes << 24);
}



void readMnistImages(const std::string& filename, std::vector<ImageData>& images) {
    std::ifstream file(filename, std::ios::binary);

    if (file.is_open()) {
        uint32_t magic_number = 0;
        uint32_t number_of_images = 0;
        uint32_t n_rows = 0;
        uint32_t n_cols = 0;

        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = reverseBytes(magic_number);
        file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
        number_of_images = reverseBytes(number_of_images);
        file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
        n_rows = reverseBytes(n_rows);
        file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
        n_cols = reverseBytes(n_cols);

        std::cout << "Trying to read image with size (" << n_rows << ", " << n_cols << ")\n";

        for (int i = 0; i < number_of_images; ++i) {

            std::vector<uint8_t> tempImage(n_rows * n_cols);
            file.read(reinterpret_cast<char*>(&tempImage[0]), n_rows * n_cols);

            //// Convert to float and normalize
            //uint8_t* normalizedImage = new uint8_t[n_rows * n_cols];
            //for (unsigned int i = 0; i < n_rows * n_cols; i++) {
            //    normalizedImage[i] = tempImage[i];
            //}

            images.push_back(ImageData(tempImage, n_cols, n_rows));
        }

        file.close();
    }
    else {
        std::cout << "Cannot open file: " << filename << std::endl;
    }
}



std::vector<ImageData> loadAndPreprocessData(const std::vector<ImageData>& originalImages, int newWidth, int newHeight) {
    
    std::vector<ImageData> newImages;
    bool progressCheck[] = { false, false, false, false, false };


    // Kiểm tra xem vector originalImages có rỗng không
    if (originalImages.empty()) {
        std::cout << "Error: No images to preprocess. 'originalImages' vector is empty.\n";
        return newImages;  // Không có dữ liệu để xử lý
    }



    // Loop through each image in the dataset
    for (int i = 0; i < originalImages.size(); i++) {

    // for (const auto& img : originalImages) {
        // Kiểm tra xem ảnh có dữ liệu hợp lệ không
        if (originalImages[i].image.empty() || (originalImages[i].width <= 0) || (originalImages[i].height <= 0)) {
            std::cout << "Warning: Encountered an invalid image with incorrect dimensions or empty data.\n";
            continue;  // Bỏ qua ảnh này và tiếp tục với ảnh tiếp theo
        }

        std::vector<uint8_t> paddedImage;

        // Apply padding if necessary
        if (originalImages[i].width != newWidth || originalImages[i].height != newHeight) {
            paddedImage = padImage(originalImages[i].image,
                originalImages[i].width, originalImages[i].height,
                newWidth, newHeight);
        }
        else {
            paddedImage = originalImages[i].image;
        }

        newImages.push_back(ImageData(paddedImage, newWidth, newHeight));

        float progress = float(i) * 5 / originalImages.size();
        if ((progress > 0) && (!progressCheck[int(progress) - 1])) {
            std::cout << "Done about " << int(progress) * 20 << "%, i = " << i << "\n";
            progressCheck[int(progress) - 1] = true;
        }
    }
    std::cout << "Finish pre-processing data\n\n";
    return newImages;
}



// Function to pad the image from 28x28 to 32x32
std::vector<uint8_t> padImage(std::vector<uint8_t> paddedImage, int oldWidth, int oldHeight, int newWidth, int newHeight) {

    std::vector<uint8_t> newImage(newWidth * newHeight);

    int paddingX = (newWidth - oldWidth) / 2;
    int paddingY = (newHeight - oldHeight) / 2;

    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            if (x >= paddingX && x < (newWidth - paddingX) && y >= paddingY && y < (newHeight - paddingY)) {
                newImage[y * newWidth + x] = paddedImage[(y - paddingY) * oldWidth + (x - paddingX)];
            }
        }
    }

    return newImage;
}






#endif