
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

		cout << "totalTime for c = a + b: " << totalTime/stoi(argv[2]) << endl;
		
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
