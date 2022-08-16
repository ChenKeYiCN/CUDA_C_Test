#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK(call)                                                              \
{                                                                               \
    const cudaError_t error = call;                                              \
    if(error != cudaSuccess)                                                    \
    {                                                                           \
        printf("[E]:%s:%d, ", __FILE__, __LINE__);                              \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));     \
        exit(1);                                                                \
    }                                                                           \
}                                                                               \

void initData(float *ptr, int size)
{
	time_t t;
	srand((unsigned int)time(&t));
	for (int i = 0; i < size; i++)
	{
		ptr[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}
__global__ void sumArrayOnGpu(float *A, float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
        C[idx] = A[idx] + B[idx];
}
void sumArrayOnHost(float *A, float *B, float *C, int n)
{
    for (int i = 0; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }
}
void checkResult(float *host_ref, float *dev_ref, int n)
{
    double epsilon = 1.0e-8;
    bool match = true;
    for(int i = 0; i < n; i++)
    {
        if(abs(host_ref[i] - dev_ref[i]) > epsilon)
        {
            match = false;
            printf("Array do not match!!!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", host_ref[i], dev_ref[i], i);
            break;
        }
    }
    if(match)
        printf("Array Match!!!!!!!!\n");
}
void printArray(float *A, int n)
{
    for(int i = 0; i < n; i++)
        printf("%f ", A[i]);
}
double cpuSeccond()
{
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}
int main()
{
    printf("start program\n");

    int dev = 0;
    cudaDeviceProp deviceprop;
    printf("start set device\n");
    CHECK(cudaGetDeviceProperties(&deviceprop, dev));
    CHECK(cudaSetDevice(dev));
    int nElem = INT_MAX / 1000;
    printf("Elem size is:\t%d\n", nElem);
    printf("start to malloc host memory\n");
    float *A, *B, *host_ref, *dev_ref;

    size_t ByteLen = sizeof(float) * nElem;
    A = (float*)malloc(ByteLen);
    B = (float*)malloc(ByteLen);
    host_ref = (float*)malloc(ByteLen);
    dev_ref = (float*)malloc(ByteLen);

    double iStart, iEnd;
    printf("start to init data\n");
    iStart = cpuSeccond();
    initData(A, nElem);
    initData(B, nElem);
    iEnd = cpuSeccond();
    printf("----Cost time in initData()\t%f\n", iEnd - iStart);

    iStart = cpuSeccond();

    memset(host_ref, 0, ByteLen);
    memset(dev_ref, 0, ByteLen);
    printf("start to sumarray on host\n");
    sumArrayOnHost(A, B, host_ref, nElem);
    iEnd = cpuSeccond();
    printf("----Cost time in sumArrayHost()\t%f\n", iEnd - iStart);



    printf("start malloc cuda memory\n");
    float *dev_A, *dev_B, *dev_C;
    iStart = cpuSeccond();
    cudaMalloc((float**)&dev_A, ByteLen);
    cudaMalloc((float**)&dev_B, ByteLen);
    cudaMalloc((float**)&dev_C, ByteLen);
    iEnd = cpuSeccond();
    printf("----Cost time in CUDAMALLOC()\t%f\n", iEnd - iStart);
    iStart = cpuSeccond();
    cudaMemcpy(dev_A, A, ByteLen, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, ByteLen, cudaMemcpyHostToDevice);
    iEnd = cpuSeccond();
    printf("----Cost time in CUDAMEMCPY()\t%f\n", iEnd - iStart);


    printf("start to generate cuda kernel");
    int thread_num = 512;
    dim3 thread_dim(thread_num);
    dim3 block_dim((nElem + thread_dim.x - 1) / thread_dim.x);
    iStart = cpuSeccond();
    sumArrayOnGpu<<<block_dim, thread_dim>>>(dev_A, dev_B, dev_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iEnd = cpuSeccond();
    printf("Execution configuration <<<%d, %d>>>---COST TIME:\t%f\n", block_dim.x, thread_dim.x, iEnd - iStart);

    printf("copy memory to host from gpu,\n");
    iStart = cpuSeccond();
    cudaMemcpy(dev_ref, dev_C, ByteLen, cudaMemcpyDeviceToHost);
    iEnd = cpuSeccond();
    printf("----Cost time in CUDAMEMCPY--DEV2HOST\t%f\n", iEnd - iStart);



    printf("start to check result!!!\n");
    checkResult(host_ref, dev_ref, nElem);

    //printArray(dev_ref, nElem);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    free(A);
    free(B);
    free(host_ref);
    free(dev_ref);

    cudaDeviceReset();
    return 0;
}