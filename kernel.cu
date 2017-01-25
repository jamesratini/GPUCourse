#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "../Timer/HighPerformanceTimer.h"

using namespace cv;
using namespace std;

int main()
{
	const unsigned int bufferSize = 1024 * 1024 * 1024;
	
	HighPrecisionTime timer; 
	ifstream largeFile("C:/Users/educ/Documents/enwiki-latest-abstract.xml");
	char* hstBuffer = nullptr;
	char* hstBitMap = nullptr;
	
	try
	{
		if (!largeFile.is_open())
			throw("File failed to Open");

		// create char array of size 1 GB
		// use constructor to initialize with 0
		hstBuffer = new char[bufferSize]();
		hstBitMap = new char[bufferSize / 8]();
		
		//read in 1GB
		timer.TimeSinceLastCall();
		largeFile.read(hstBuffer, bufferSize);

		if (!largeFile)
			throw("Failed to read into buffer");

		printf("Success! It took %f seconds to read the file \n", timer.TimeSinceLastCall());
		
	}
	catch (char* error)
	{
		printf("%s", error);
		goto bad_exit;
	}

	//clean up
bad_exit:
	if (largeFile.is_open())
		largeFile.close();
	if (hstBuffer)
		delete[] hstBuffer;
	if (hstBitMap)
		delete[] hstBitMap;


	system("pause");
    return 0;
}

