#include <stdlib.h>
#include <string.h>
#include <time.h>

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
		printf("%d\n", ptr[i]);
}
int main()
{
	int nElem = 1024;
	size_t nBytes = nElem * sizeof(float);
	float *A, *B, *C;
	A = (float*)malloc(nBytes);
	B = (float*)malloc(nBytes);
	C = (float*)malloc(nBytes);

	initData(A, nElem);
	initData(B, nElem);
	sumArrayOnHost(A, B, C, nElem);
	printArray(C, nElem);
	free(A);
	free(B);
	free(C);

	return 0;
}