# CUDA_C_Test

# ch2问题
###	1.为什么多线程时，核函数打印的值和最终结果似乎不一样
### 2.__global__<<<3,5>>>到底组织了怎么样的线程形式?看起去是一维的，因为blockIdx.y, blockIdx.z, threadIdx.y, threadIdx.z都为0.