
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
unsigned char* BoxFilter(unsigned char* src, unsigned char* dst, int imgW, int imgH, unsigned char* kernelArray, int kW, int kH, unsigned char* temp);

int main(int argc, char** argv)
{
	Mat hostImage;
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

	imgWidth = hostImage.cols;
	imgHeight = hostImage.rows;
	imageDst = hostImage.data;
	temp = hostImage.data;

	temp = BoxFilter(hostImage.data, imageDst, imgWidth, imgHeight, K, 3, 3, temp);
	hostImage.data = temp;
	
	namedWindow("Display window", WINDOW_NORMAL);
	resizeWindow("Display window", 1900, 1080);
	imshow("Display window", hostImage);

	waitKey(0);
	return 0;
}

unsigned char* BoxFilter(unsigned char* src, unsigned char* dst, int imgW, int imgH, unsigned char* kernelArray, int kW, int kH, unsigned char* temp)
{
	int currentPixel = 0;
	unsigned char sumOfColor = 0;
	for (int i = 0; i < imgW; i++)
	{
		for (int j = 0; j < imgH; j++)
		{
			//identify our current pixel
			currentPixel = i * j;
			int count = 0;
			//for every neighboring pixel within radius in the x direction
			for (int k = 0; k < kW; k++)
			{
				//for everything neighboring pixel within radius in the y direction
				for (int l = 0; l < kH; l++)
				{
					sumOfColor += src[currentPixel + k + l] * kernelArray[count];
					count++;
				}
			}
			//divide sum by kernel.size() and store at dst[currentPixel]
			dst[currentPixel] = sumOfColor / (kW*kH);
		}
	}

	return dst;
}