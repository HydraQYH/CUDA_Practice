#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
const int BLOCK = 4096;
const int WARP_SIZE = 32;

__global__ void kernel_fast(float* p)
{ 
    __shared__ float data[WARP_SIZE][WARP_SIZE];    // 共占用4KB的共享内存
    // 获取当前线程属于第几个线程束
    int warp_id = threadIdx.y;
    int thread_id = threadIdx.x;
    // 从全局内存读取数据到共享内存
    // int offset = blockIdx.x * gridDim.x + threadIdx.y * blockDim.y + threadIdx.x;
    // data[warp_id][thread_id] = p[offset];
    data[warp_id][thread_id] *= 2;  // 同一个线程束中的相邻的线程访问相邻的元素 相邻的元素处于不同的bank
}

__global__ void kernel_slow(float* p)
{ 
    __shared__ float data[WARP_SIZE][WARP_SIZE];
    int warp_id = threadIdx.y;
    int thread_id = threadIdx.x;
    // int offset = blockIdx.x * gridDim.x + threadIdx.y * blockDim.y + threadIdx.x;
    // data[thread_id][warp_id] = p[offset];
    data[thread_id][warp_id] *= 2;
}

int main(void)
{
    float* d_data = nullptr;
    cudaMalloc((void**)&d_data, BLOCK * WARP_SIZE * WARP_SIZE * sizeof(float));
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    dim3 grid(BLOCK);
    dim3 block(WARP_SIZE, WARP_SIZE);
    
    cudaEventRecord(start);
    for (size_t i = 0; i < 1000; i++)
    {
        kernel_slow<<<grid, block>>>(d_data);
        // kernel_fast<<<grid, block>>>(d_data);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    std::cout << "Elapsed Time: " << elapsed_time << " ms" << std::endl;
    
    cudaEventDestroy(end);
    cudaEventDestroy(start);
    cudaFree(d_data);
    return 0;
}

