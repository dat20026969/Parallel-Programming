#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <float.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <curand_kernel.h>


#include "ConvolutionDevice.cuh"
#include "FileHandler.h"
#include "Properties.cuh"
#include "MainCompiler.cuh"
#include "MainCompilerWithStream.cuh"
#include "MainCompilerWhole.cuh"



void printDeviceInfo() {
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
    printf("****************************\n");
}




int main(int argc, char** argv) {

    printDeviceInfo();
	
	int imageId = 0;
	int compilingType = 0;
	bool saveKernel = false;
	bool saveImage = false;
	int blockSize1 = 16;
	int blockSize3 = 8;
	
	MainCompiler compiler;
	
	std::cout << "\nAttempting to read file\n";
    compiler.readFile();
	
	if (argc > 1) {
		// Syntax: ./a.out oneImage
		//                 [image type] [compiling type]
		//                 [save/load kernel] [save/not save image]
		//                 [layer-1-&-2 block size] [layer-3-&-4 block size]
		if (strcmp(argv[1], "oneImage") == 0) {
			
			if (argc > 2)
				imageId = atoi(argv[2]);
			else
				std::cout << "The default image ID is " << imageId << "\n";
			
			
			// [compiling type]. If number is 0 - 3: run one kernel. if number is -1: run all kernels.
			if (argc > 3)
				compilingType = atoi(argv[3]);
			else
				std::cout << "The default compiling type is " << compilingType << "\n";


			// [save/load kernel]. 
			if ((argc > 4) && ((argv[4][0] == 'S') || (argv[4][0] == 's')))
				saveKernel = true;
			else
				std::cout << (saveKernel ? "Save the kernels" : "Load the kernels") << " by default\n";


			// [save/not save image]
			if ((argc > 5) && ((argv[5][0] == 'Y') || (argv[5][0] == 'y')))
				saveImage = true;
			else
				std::cout << (saveImage ? "Save the image in CSVs" : "Print the image") << " by default\n";


			// [layer-1-&-2 block size]. Block size for layer 1 and 2
			if (argc > 6)
				blockSize1 = atoi(argv[6]);
			else
				std::cout << "The default block size for layer 1 and layer 2 is " << blockSize1 << "\n";


			// [layer-3-&-4 block size]. Block size for layer 3 and 4
			if (argc > 7)
				blockSize3 = atoi(argv[7]);
			else
				std::cout << "The default block size for layer 3 and layer 4 is " << blockSize3 << "\n";


			std::cout << "\nAttempting to assign host memory\n";
			compiler.assignHostMemory();


			if (!saveKernel) {
				std::cout << "\nAttempting to load kernels\n";
				compiler.loadKernel();
			}
			else {
				std::cout << "\nAttempting to generate kernels\n";
				compiler.generateKernel();
			}


			switch (compilingType) {

			// Run on all functions
			case -1:
				std::cout << "\nAttempting to run all functions\n";
				compiler.assignDeviceMemory();
				compiler.runAll(imageId, blockSize1, blockSize3);
				break;

			// Run on host
			case 0:
				std::cout << "\nAttempting to run on host\n";
				compiler.runOnHost(imageId);
				break;

			// Run on device
			case 1: case 2: case 3:
				std::cout << "\nAttempting to assign device memory\n";
				compiler.assignDeviceMemory();
				compiler.runOnDevice(imageId, compilingType, blockSize1, blockSize3);
				break;
			}


			if (saveKernel) {
				std::cout << "\nAttempting to save kernels\n";
				compiler.saveKernel();
			}


			if (saveImage && (imageId != -1) && (compilingType != -1)) {
				std::cout << "\nAttempting to save a file\n";
				compiler.saveRawText();
			}
		}
		
		// Syntax: ./a.out withStream
		//                 [layer-1-&-2 block size] [layer-3-&-4 block size]
		else if (strcmp(argv[1], "withStream") == 0) {
			
			MainCompilerWithStream temp;
			temp.copyFile(compiler.trainData, compiler.testData);
			
			
			// [layer-1-&-2 block size]. Block size for layer 1 and 2
			if (argc > 2)
				blockSize1 = atoi(argv[2]);
			else
				std::cout << "The default block size for layer 1 and layer 2 is " << blockSize1 << "\n";


			// [layer-3-&-4 block size]. Block size for layer 3 and 4
			if (argc > 3)
				blockSize3 = atoi(argv[3]);
			else
				std::cout << "The default block size for layer 3 and layer 4 is " << blockSize3 << "\n";
			
			
			std::cout << "\nAttempting to load the kernel\n";
			temp.loadKernel();

			std::cout << "\nAttempting to assign host memory\n";
			temp.assignHostMemory();
			
			std::cout << "\nAttempting to assign device memory\n";
			temp.assignDeviceMemory();
			
			std::cout << "\nAttempting to run on device memory\n";
			temp.runWithStream(blockSize1, blockSize3);
		}
		
		// Syntax: ./a.out biggerImage
		//                 [layer-1-&-2 block size] [layer-3-&-4 block size]
		else if (strcmp(argv[1], "biggerImage") == 0) {
			
			MainCompilerWhole temp;
			temp.copyFile(compiler.trainData, compiler.testData);
			
			// [layer-1-&-2 block size]. Block size for layer 1 and 2
			if (argc > 2)
				blockSize1 = atoi(argv[2]);
			else
				std::cout << "The default block size for layer 1 and layer 2 is " << blockSize1 << "\n";


			// [layer-3-&-4 block size]. Block size for layer 3 and 4
			if (argc > 3)
				blockSize3 = atoi(argv[3]);
			else
				std::cout << "The default block size for layer 3 and layer 4 is " << blockSize3 << "\n";
			
			std::cout << "\nAttempting to load the kernel\n";
			temp.loadKernel();

			std::cout << "\nAttempting to assign host memory\n";
			temp.assignHostMemory();

			std::cout << "\nAttempting to assign a new kind of input data\n";
			temp.assignInputData();

			std::cout << "\nAttempting to assign device memory\n";
			temp.assignDeviceMemory();

			std::cout << "\nAttempting to run the kernel\n";
			temp.runWithBiggerImage(true, blockSize1, blockSize3);

			temp.printFirstResult();
		}
	}
	
	
    std::cout << "\nFinish!\n";


    return 0;
}
