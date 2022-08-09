#include <stdio.h>
__global__ void hellofromGPU(void)
{
	if(threadIdx.x > 5)
		printf("hello from GPU, thread is %d\n", threadIdx.x);
}

int main()
{
	printf("hello CUDA, Im coming!\n");
	hellofromGPU<<<5, 10>>>();
	cudaDeviceReset();
	return 0;
}

