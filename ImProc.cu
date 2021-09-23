#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring>
#include <iostream>

#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char* file, const int line)
{
	if (cudaSuccess != err)
	{
		std::cerr << file << "(" << line << ") : CUDA Runtime API error " << err << ": " << cudaGetErrorString(err) << ".\n";
		exit(EXIT_FAILURE);
	}
}

__global__ void kernel(cudaTextureObject_t tex, int width, int height, unsigned char* outputData)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	outputData[y * width * 3 + 3 * x] = tex2D<unsigned char>(tex, 3 * x, y);
	outputData[y * width * 3 + 3 * x + 1] = tex2D<unsigned char>(tex, 3 * x + 1, y);
	outputData[y * width * 3 + 3 * x + 2] = tex2D<unsigned char>(tex, 3 * x + 2, y);
}

extern "C" void process(unsigned char* inBuffer, int width, int height, int channels, unsigned char** outBuffer, int &stride)
{
	size_t inputStride = sizeof(unsigned char) * width * channels;

	unsigned char* devImageIn = nullptr;
	size_t inPitch;
	cudaError_t err = cudaMallocPitch(&devImageIn, &inPitch, inputStride, height);
	err = cudaMemcpy2D(devImageIn, inPitch, inBuffer, inputStride, inputStride, height, cudaMemcpyHostToDevice);
	checkCudaErrors(err);

	unsigned char* devImageOut = nullptr;
	size_t outPitch;
	err = cudaMallocPitch(&devImageOut, &outPitch, inputStride, height);
	checkCudaErrors(err);

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypePitch2D;
	texRes.res.pitch2D.devPtr = devImageIn;
	texRes.res.pitch2D.desc = desc;
	texRes.res.pitch2D.width = static_cast<size_t>(width) * channels;
	texRes.res.pitch2D.height = height;
	texRes.res.pitch2D.pitchInBytes = inPitch;
	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeWrap;
	texDescr.addressMode[1] = cudaAddressModeWrap;
	texDescr.readMode = cudaReadModeElementType;

	cudaTextureObject_t texture;
	err = cudaCreateTextureObject(&texture, &texRes, &texDescr, NULL);
	checkCudaErrors(err);

	dim3 blockSize(16, 16);
	dim3 gridSize(width + blockSize.x / blockSize.x, height + blockSize.y/ blockSize.y);
	
	kernel<<<gridSize, blockSize>>>(texture, width, height, devImageOut);
	stride = static_cast<int>(outPitch);
	*outBuffer = new unsigned char[stride * height];
	for (int i = 0; i < stride * height; i += 3)
	{
		(*outBuffer)[i] = 255;
		(*outBuffer)[i+1] = 0;
		(*outBuffer)[i+2] = 0;
	}
	err = cudaMemcpy2D(*outBuffer, inPitch, devImageOut, inputStride, inputStride, height, cudaMemcpyDeviceToHost);
	checkCudaErrors(err);

	cudaDestroyTextureObject(texture);
	cudaFree(devImageIn);
	cudaFree(&inPitch);
	cudaFree(&outPitch);
}
