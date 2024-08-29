#ifndef IMAGE_DATA_
#define IMAGE_DATA_



#include <vector>



// Structure for 2D images
struct ImageData {
    std::vector<uint8_t> image; // Store as a 1D array
    int width;
    int height;
    int label;

    ImageData(std::vector<uint8_t> img, int w, int h) : image(img), width(w), height(h), label(-1) {}
};



#endif