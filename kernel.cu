#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <stdlib.h>
#include <ctime>
#include "../HighPerformanceTimer/HighPerformanceTimer.h"
#include <omp.h>

using namespace std;

typedef int ArrayType_t;

bool arrayMalloc(ArrayType_t** a, ArrayType_t** b, ArrayType_t** c, int size);
void arrayFree(ArrayType_t* a, ArrayType_t* b, ArrayType_t* c);
void arrayInit(ArrayType_t* a, ArrayType_t* b, ArrayType_t* c, int size);
void addCPUVec(ArrayType_t* a, ArrayType_t* b, ArrayType_t* c, int size);
cudaError_t addWithCuda(ArrayType_t *c, const ArrayType_t *a, const ArrayType_t *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
int main(int argc, char* argv[])
{
	int size = 10;

	//check if argv[1] is valid
	if (argc > 1)
	{
		size = stoi(argv[1]);
	}

	ArrayType_t* a = nullptr;
	ArrayType_t* b = nullptr;
	ArrayType_t* c = nullptr;

	try
	{
		//allocate and verify success
		if (!arrayMalloc(&a, &b, &c, size))
		{
			throw("Allocation Failed");
		}

		HighPrecisionTime timer;
		double totalTime = 0.0;


		arrayInit(a, b, c, size);

		if (argv[2] == NULL)
		{
			argv[2] = "100";
		}
		for (int i = 0; i < stoi(argv[2]); i++)
		{
			timer.TimeSinceLastCall();
			addCPUVec(a, b, c, size);
			totalTime += timer.TimeSinceLastCall();
		}

		cout << "totalTime for c = a + b: " << totalTime / stoi(argv[2]) << endl;

	}
	catch (char* error)
	{
		cout << "An Exception Occured: " << error << endl;
	}

	arrayFree(a, b, c);


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
cudaError_t addWithCuda(ArrayType_t *c, const ArrayType_t *a, const ArrayType_t *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;
	int arraySize = size * sizeof(ArrayType_t);
	HighPrecisionTime timer;
	double totalTime = 0.0;

	try
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw("setDevice Failed");
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

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, a, arraySize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw("copy of a failed");
		}

		cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			throw("copy of b failed");
		}

		// Launch a kernel on the GPU with one thread for each element.
		timer.TimeSinceLastCall();
		addKernel << <1, size >> >(dev_c, dev_a, dev_b);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			throw("addKernel failed");
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			throw("syncronize failed");
		}
		totalTime += timer.TimeSinceLastCall();

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(c, dev_c, arraySize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw("copy from device c to host c failed");
		}
	}
	catch (char* error)
	{
		printf("Error Message: %c", error);
		goto Error;
	}
	

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
