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

void Threshold(Mat image, unsigned char threshold);
void thresholdWithCuda(Mat* image, unsigned char threshold);

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

	imshow("Display window", hostImage);
}

int main(int argc, char** argv)
{

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

	thresholdWithCuda(&hostImage, THRESHOLD);

	namedWindow("Display window", WINDOW_NORMAL);
	resizeWindow("Display window", 1900, 1080);
	createTrackbar("Threshold", "Display window", &thresholdSlider, THRESHOLD_SLIDER_MAX, on_Trackbar);
	imshow("Display window", hostImage);

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