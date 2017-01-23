#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

//GLOBALS
int thresholdSlider;
const int THRESHOLD_SLIDER_MAX = 255;
cudaDeviceProp deviceProps;
Mat hostImage;
unsigned char THRESHOLD = 120;
unsigned char *dev0_image;
unsigned char *devCopy_image;
int imageSize = 0;
//GLOBALS

__global__ void kernel(unsigned char* imageOrig, unsigned char* imageCopy, unsigned char threshold)
{

	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (imageOrig[j] > threshold)
	{
		imageCopy[j] = 255;
	}
	else
	{
		imageCopy[j] = 0;
	}
}
__global__ void convolutionGPU(unsigned char* imageOrig, unsigned char* imageCopy, int imgW, int imgH, unsigned char* kernelArray, int kW, int kH)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int kernel_size, half_kernel_size;
	kernel_size = 3;
	half_kernel_size = kernel_size / 2;
	float sumOfColor = 0.0;
	int count = 0;

	//go through the rows
	for (int i = -half_kernel_size; i < half_kernel_size; i++ )
	{
		if (i + y < 0 || i + y >= imgH)
			continue;

		sumOfColor += *(imageOrig + (i + y) * imgW + x) * kernelArray[count];
		for (int j = 1; j < kW / 2; j++)
		{
			if (x - j >= 0)
			{
				sumOfColor += *(imageOrig + (i + y) * imgW - i + x) * kernelArray[count];
			}
			if (x + j < imgW)
			{
				sumOfColor += *(imageOrig + (i + y) * imgW + i + x) * kernelArray[count];
			}
			count++;
		}
		
	}
	*(imageCopy + y * imgW + x) = (unsigned char)(sumOfColor / (kW * kH));
}

void Threshold(Mat image, unsigned char threshold);
void thresholdWithCuda(Mat* image, unsigned char threshold);
unsigned char* BoxFilter(unsigned char* src, unsigned char* dst, int imgW, int imgH, unsigned char* kernelArray, int kW, int kH, unsigned char* temp);
void BoxFilterGPU(Mat* hostImage, unsigned char* src, unsigned char* dst, int imgW, int imgH, unsigned char* kernelArray, int kW, int kH);

void on_Trackbar(int, void *)
{
	int blocksNeeded = (imageSize + deviceProps.maxThreadsPerBlock - 1) / deviceProps.maxThreadsPerBlock;
	//use kernel to threshold dev0_image, then write to devCopy_image
	kernel << <blocksNeeded, deviceProps.maxThreadsPerBlock >> >(dev0_image, devCopy_image, thresholdSlider);
	cudaDeviceSynchronize();

	if (cudaMemcpy(hostImage.data, devCopy_image, imageSize, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		throw("trackbar memcopy failed");
	}

	imshow("Display", hostImage);
}

int main(int argc, char** argv)
{
	unsigned char* imageDst;
	unsigned char* temp;
	unsigned char K[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	int imgWidth = 0;
	int imgHeight = 0;

	//if 2 arguements arent passed, tell user there was an error
	if (argc != 2) {
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	//set hostImage to command argument
	hostImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if (!hostImage.data) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	//display image information
	cout << "Image has: " << hostImage.channels() << " channels" << endl;
	cout << "Image is " << hostImage.cols << "x" << hostImage.rows << endl;

	//convert image to grayscale
	cvtColor(hostImage, hostImage, cv::COLOR_RGB2GRAY);

	//thresholdWithCuda(&hostImage, THRESHOLD);

	imgWidth = hostImage.cols;
	imgHeight = hostImage.rows;

	imageDst = hostImage.data;
	temp = hostImage.data;

	// create window and display original image
	namedWindow("Display", WINDOW_NORMAL);
	resizeWindow("Display", 1900, 1080);
	//createTrackbar("Threshold", "Display", &thresholdSlider, THRESHOLD_SLIDER_MAX, on_Trackbar);
	imshow("Display", hostImage);
	waitKey(0);

	//temp = BoxFilter(hostImage.data, imageDst, imgWidth, imgHeight, K, 3, 3, temp);
	BoxFilterGPU(&hostImage, dev0_image, devCopy_image, imgWidth, imgHeight, K, 3, 3);

	//hostImage.data = temp;

	//display new blurred image
	imshow("Display", hostImage);

	waitKey(0);

	return 0;
}

void Threshold(Mat hostImage, unsigned char threshold)
{
	int height = hostImage.rows;
	int width = hostImage.cols;

	for (int i = 0; i < height*width; i++)
	{
		if (hostImage.data[i] > threshold)
		{
			hostImage.data[i] = 255;
		}
		else
		{
			hostImage.data[i] = 0;
		}
	}
}
void BoxFilterGPU(Mat* hostImage, unsigned char* src, unsigned char* dst, int imgW, int imgH, unsigned char* kernelArray, int kW, int kH)
{
	cudaError_t cudaStatus;
	int imageSize = imgW * imgH;

	try
	{
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess)
		{
			throw("error in set device");
		}

		cudaStatus = cudaGetDeviceProperties(&deviceProps, 0);
		if (cudaStatus != cudaSuccess) {
			throw("getDeviceProperties failed");
		}

		cudaStatus = cudaMalloc((void**)&src, imageSize);
		if (cudaStatus != cudaSuccess)
		{
			throw("cudaMalloc failed");
		}

		cudaStatus = cudaMalloc((void**)&dst, imageSize);
		if (cudaStatus != cudaSuccess)
		{
			throw("cudaMalloc failed");
		}

		cudaStatus = cudaMemcpy(src, hostImage->data, imageSize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			throw("mem copy failed 1");
		}

		int blocksNeeded = (imageSize + deviceProps.maxThreadsPerBlock - 1) / deviceProps.maxThreadsPerBlock;

		convolutionGPU << <blocksNeeded, deviceProps.maxThreadsPerBlock >> >(src, dst, imgW, imgH, kernelArray, kW, kH);
		if (cudaGetLastError() != cudaSuccess)
			throw("add Kernel failed");

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			throw("device sync failed");
			
		}

		cudaStatus = cudaMemcpy(hostImage->data, dst, imageSize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			throw("mem copy failed 2");
		}
	}
	catch (char* error)
	{
		cout << error << endl;
		
		goto bad_exit;
	}
bad_exit:
	cudaFree((void*)&dev0_image);
	cudaFree((void*)&devCopy_image);
	
}
void thresholdWithCuda(Mat* hostImage, unsigned char threshold)
{
	cudaError_t cudaStatus;

	int height = hostImage->rows;
	int width = hostImage->cols;
	imageSize = height * width;

	try
	{
		//set device
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess)
		{
			throw("error in set device");
		}
		//check device props
		cudaStatus = cudaGetDeviceProperties(&deviceProps, 0);
		if (cudaStatus != cudaSuccess) {
			throw("getDeviceProperties failed");
		}
		//malloc original image and copy image
		cudaStatus = cudaMalloc((void**)&dev0_image, imageSize);
		if (cudaStatus != cudaSuccess)
		{
			throw("cudaMalloc failed");
		}

		cudaStatus = cudaMalloc((void**)&devCopy_image, imageSize);
		if (cudaStatus != cudaSuccess)
		{
			throw("cudaMalloc failed");
		}

		//copy original to gpu
		cudaStatus = cudaMemcpy(dev0_image, hostImage->data, imageSize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			throw("mem copy failed");
		}

		int blocksNeeded = (imageSize + deviceProps.maxThreadsPerBlock - 1) / deviceProps.maxThreadsPerBlock;
		//use kernel to threshold dev0_image, then write to devCopy_image
		kernel << <blocksNeeded, deviceProps.maxThreadsPerBlock >> >(dev0_image, devCopy_image, threshold);
		if (cudaGetLastError() != cudaSuccess)
			throw("add Kernel failed");
		cudaStatus = cudaDeviceSynchronize();

		cudaStatus = cudaMemcpy(hostImage->data, devCopy_image, imageSize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			throw("mem copy failed");
		}
	}
	catch (char* error)
	{
		cout << error << endl;
		goto bad_exit;
	}
bad_exit:
	cudaFree((void*)&dev0_image);
	cudaFree((void*)&devCopy_image);
}
unsigned char* BoxFilter(unsigned char* src, unsigned char* dst, int imgW, int imgH, unsigned char* kernelArray, int kW, int kH, unsigned char* temp)
{
	float sumOfColor = 0;
	for (int x = 1; x < imgW - 1; x++)
	{
		for (int y = 1; y < imgH - 1; y++)
		{
			sumOfColor = 0;
			int count = 0;
			//for every neighboring pixel within radius in the x direction
			for (int k = -(kW/2); k <= kW/2; k++)
			{
				//for everything neighboring pixel within radius in the y direction
				for (int l = -(kH/2); l <= kH/2; l++)
				{
					sumOfColor += src[(y + l)*imgW + (x + k)] * kernelArray[count];
					count++;
				}
			}
			*(dst + y * imgW + x) = (unsigned char)(sumOfColor / (kW*kH));
		}
	}

	return dst;
}