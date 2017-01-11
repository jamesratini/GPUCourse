#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <ctime>
#include "../HighPerformanceTimer/HighPerformanceTimer.h"


using namespace std;

typedef int ArrayType_t;

bool arrayMalloc(ArrayType_t** a, ArrayType_t** b, ArrayType_t** c, int size);
void arrayFree(ArrayType_t* a, ArrayType_t* b, ArrayType_t* c);
void arrayInit(ArrayType_t* a, ArrayType_t* b, ArrayType_t* c, int size);
void addCPUVec(ArrayType_t* a, ArrayType_t* b, ArrayType_t* c, int size);
cudaError_t addWithCuda(ArrayType_t *c, const ArrayType_t *a, const ArrayType_t *b, unsigned int size, int repCount);

__global__ void addKernel(ArrayType_t *c, const ArrayType_t *a, const ArrayType_t *b, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		c[i] = a[i] + b[i];
	}

}
int main(int argc, char* argv[])
{
	int size = 10;

	//check if argv[1] is valid
	if (argc > 1)
	{
		size = stoi(argv[1]);
	}

	if (argv[2] == NULL)
	{
		argv[2] = "100";
	}

	ArrayType_t* a = nullptr;
	ArrayType_t* b = nullptr;
	ArrayType_t* c = nullptr;
	vector<int> sizes = { 100, 1000, 10000, 100000, 1000000 };
	vector<int> reps = { 10, 100, 1000 };

	

	try
	{
		//allocate and verify success
		

		HighPrecisionTime timer;
		double cpuTime = 0.0;

		for each (int k in sizes)
			for each (int repCount in reps) {
				printf("---------------------");
				cout << "Array Size: " << k << endl;
				cout << "Rep Count: " << repCount << endl;

				if (!arrayMalloc(&a, &b, &c, k))
				{
					throw("Allocation Failed");
				}
				cout << "here" << endl;
				//arrayInit(a, b, c, k);
				srand(time(NULL));

				for (int i = 0; i < size; i++)
				{
					//cout << i << endl;
					//fill a with random numbers
					a[i] = (rand() % size) + 1;
					//fill b with random numbers
					b[i] = (rand() % size) + 1;
					//fill c with 0s
					c[i] = 1;
				}


				for (int i = 0; i < repCount; i++)
				{
					timer.TimeSinceLastCall();
					addCPUVec(a, b, c, size);
					cpuTime += timer.TimeSinceLastCall();
				}

				cout << "Average time on cpu to compute c = a + b: " << cpuTime / repCount << endl;

				addWithCuda(c, a, b, k, repCount);

				arrayFree(a, b, c);

				printf("------------------- \n");
				

			}

	}
	catch (char* error)
	{
		cout << "An Exception Occured: " << error << endl;
	}

#ifdef _WIN32 || _WIN64
	system("pause");
#endif
	


	return 0;
}

bool arrayMalloc(ArrayType_t** a, ArrayType_t** b, ArrayType_t** c, int size)
{
	bool retVal = false;
	*a = (ArrayType_t*)malloc(size * sizeof(ArrayType_t));
	*b = (ArrayType_t*)malloc(size * sizeof(ArrayType_t));
	*c = (ArrayType_t*)malloc(size * sizeof(ArrayType_t));

	if (*a != NULL && *b != NULL && *c != NULL)
	{
		retVal = true;
	}

	return retVal;
}
void arrayFree(ArrayType_t* a, ArrayType_t* b, ArrayType_t* c)
{
	free(a);
	free(b);
	free(c);
}
void arrayInit(ArrayType_t* a, ArrayType_t* b, ArrayType_t* c, int size)
{
	srand(time(NULL));

	for (int i = 0; i < size; i++)
	{
		cout << i << endl;
		//fill a with random numbers
		a[i] = (rand() % size) + 1;
		//fill b with random numbers
		b[i] = (rand() % size) + 1;
		//fill c with 0s
		c[i] = 1;
	}

}
void addCPUVec(ArrayType_t* a, ArrayType_t* b, ArrayType_t* c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}
cudaError_t addWithCuda(ArrayType_t *c, const ArrayType_t *a, const ArrayType_t *b, unsigned int size, int repCount)
{
	ArrayType_t *dev_a = 0;
	ArrayType_t *dev_b = 0;
	ArrayType_t *dev_c = 0;
	cudaError_t cudaStatus;
	int arraySize = size * sizeof(ArrayType_t);
	HighPrecisionTime timer;
	double gpuTime = 0.0;
	double memCopyTime = 0.0;
	double memCCopy = 0.0;
	cudaDeviceProp deviceProps;

	

	try
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw("setDevice Failed");
		}
		//check device props
		cudaStatus = cudaGetDeviceProperties(&deviceProps, 0);
		if (cudaStatus != cudaSuccess){
			throw("getDeviceProperties failed");
		}

		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_c, arraySize);
		if (cudaStatus != cudaSuccess) {
			throw("malloc of c failed");
		}

		cudaStatus = cudaMalloc((void**)&dev_a, arraySize);
		if (cudaStatus != cudaSuccess) {
			throw("malloc of a failed");
		}

		cudaStatus = cudaMalloc((void**)&dev_b, arraySize);
		if (cudaStatus != cudaSuccess) {
			throw("malloc of b failed");
		}

		//time memory copy
		timer.TimeSinceLastCall();
		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, a, arraySize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw("copy of a failed");
		}

		cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw("copy of b failed");
		}
		
		memCopyTime = timer.TimeSinceLastCall();
		cout << "Memory copy took: " << memCopyTime << endl;

		int blocksNeeded = (size + deviceProps.maxThreadsPerBlock - 1) / deviceProps.maxThreadsPerBlock;
		cout << "Will launch " << blocksNeeded << " block(s) of " << deviceProps.maxThreadsPerBlock << " threads" << endl;

		//loop through reps, addKernel everytime
		
		for (int i = 0; i < repCount; i++)
		{
			timer.TimeSinceLastCall();
			addKernel << < blocksNeeded, deviceProps.maxThreadsPerBlock >> >(dev_c, dev_a, dev_b, size);
			if (cudaGetLastError() != cudaSuccess)
				throw("add Kernel failed");
			cudaStatus = cudaDeviceSynchronize();
			gpuTime += timer.TimeSinceLastCall();

		}
			

			

		

		cout << "Average time for GPU to compute: " << gpuTime / repCount << endl;

		//time last copy
		timer.TimeSinceLastCall();
		cudaStatus = cudaMemcpy(c, dev_c, arraySize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw("copy from device c to host c failed");
		}
		memCCopy = timer.TimeSinceLastCall();
		cout << "Time to move c from device to host: " << memCCopy << endl;

		double grandTotal = memCCopy + gpuTime + memCopyTime;
		
		cout << "Grand total time spent in gpu: " << grandTotal << endl;

	}
	catch (char* error)
	{
		cout << "error message: " << error << endl;
		goto Error;
	}


Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
