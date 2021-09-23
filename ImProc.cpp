#include "ImProc.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

extern "C" void process(unsigned char* inBuffer, int width, int height, int channels, unsigned char** outBuffer, int& stride);

int main(int argc, char** argv)
{
	if (argc != 3)
	{
		std::cout << "Usage: " << argv[0] << " [input file] [output file]\n";
		exit(1);
	}

	const char* inputPath = argv[1];
	const char* outputPath = argv[2];

	int width;
	int height;
	int channels;
	unsigned char* inputImage = stbi_load(inputPath, &width, &height, &channels, 0);
	if (inputImage == nullptr)
	{
		std::cout << "Cannot load image\n";
		exit(1);
	}

	unsigned char* outputImage = nullptr;
	int stride;
	process(inputImage, width, height, channels, &outputImage, stride);

	stbi_write_png(outputPath, width, height, channels, outputImage, stride);

	delete[] outputImage;
	stbi_image_free(inputImage);

	return 0;
}
