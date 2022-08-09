#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>


__global__ void sumArrayOnDevice(float *A, float *B, float *C, int n)
{
	for (int i = 0; i < n; i++)
	{
		C[i] = A[i] + B[i];
		printf("%f\n", C[i]);
	}
}

void sumArrayOnHost(float *A, float *B, float *C, int n)
{
	for (int i = 0; i < n; i++)
	{
		C[i] = A[i] + B[i];
	}
}

void initData(float *ptr, int size)
{
	time_t t;
	srand((unsigned int)time(&t));
	for (int i = 0; i < size; i++)
	{
		ptr[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}
void printArray(float* ptr, int size)
{
	for (int i = 0; i < size; i++)
		printf("%f ", ptr[i]);
}
int main()
{
	int nElem = 1024;
	size_t nBytes = nElem * sizeof(float);
	float *A, *B, *C;
	A = (float*)malloc(nBytes);
	B = (float*)malloc(nBytes);
	C = (float*)malloc(nBytes);

	float *cA, *cB, *cC;
	cudaMalloc((float**)&cA, nBytes);
	cudaMalloc((float**)&cB, nBytes);
	cudaMalloc((float**)&cC, nBytes);

	initData(A, nElem);
	initData(B, nElem);

	cudaMemcpy(cA, A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(cB, A, nBytes, cudaMemcpyHostToDevice);

	//sumArrayOnHost(A,B,C,nElem);
	
	sumArrayOnDevice<<<1,1>>>(cA, cB, cC, nElem);
	printArray(C, nElem);
	printf("\n\n\n\n\n");

	//	block, wait for answer
	cudaMemcpy(C, cC, nBytes, cudaMemcpyDeviceToHost);
	
	// maybe gpu not finish,TODO!!!
	printArray(C, nElem);
	free(A);
	free(B);
	free(C);

	cudaFree(cA);
	cudaFree(cB);
	cudaFree(cC);

	return 0;
}