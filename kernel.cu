
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

void Threshold(Mat image, unsigned char threshold);

int main(int argc, char** argv)
{
	//if 2 arguements arent passed, tell user there was an error
	if (argc != 2){
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	//create Mat and attempt to read image into it from argv[1]
	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if (!image.data) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}
	
	cout << "Image has: " << image.channels() << " channels" << endl;
	cout << "Image is " << image.cols << "x" << image.rows << endl;

	cvtColor(image, image, cv::COLOR_RGB2GRAY);
	cout << "Converted to gray" << endl;

	unsigned char THRESHOLD = 200;
	Threshold(image, THRESHOLD);

	namedWindow("Dispaly window", WINDOW_NORMAL);
	imshow("Display window", image);

	//resizeWindow("Display window", 1920, 1080);

	waitKey(0);
	return 0;
}

void Threshold(Mat image, unsigned char threshold)
{
	int height = image.rows;
	int width = image.cols;

	for (int i = 0; i < height*width; i++)
	{
		if (image.data[i] > threshold)
		{
			image.data[i] = 255;
		}
		else
		{
			image.data[i] = 0;
		}
	}
}