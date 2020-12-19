#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#define N 8192
#define DIM 33
#define OUTPUT_DIM 31

void printMatrix(float* p, int row, int column)
{
    for (int i = 0; i < row; i++)
    {
        printf("[ ");
        for (int j = 0; j < column; j++)
        {
            if (j != column - 1)
            {
                printf("%.3f, ", *(p + i * column + j));
            }
            else
            {
                printf("%.3f ]\n", *(p + i * column + j));
            }
        }
    }
}

__global__ void AveragePooling2D(float* p_value, float* p_result)
{
    // pos为输出中的位置
    int pos = threadIdx.y * blockDim.x + threadIdx.x;
    // 计算对应输入区域的中心位置
    int x = threadIdx.x + 1;
    int y = threadIdx.y + 1;
    
    float result = 0;
    int start = OUTPUT_DIM / 2 * (-1); 
    int stop = OUTPUT_DIM / 2;
    for (int i = start; i <= stop; i++)
    {
        for (int j = start; j <= stop; j++)
        {
            int x_ = x + i;
            int y_ = y + j;
            result += *(p_value + y_ * DIM + x_);
        }
    }
    result /= (OUTPUT_DIM * OUTPUT_DIM);
    *(p_result + pos) = result;
}

int main(void)
{
    float* host_p;
    float* host_r;
    float* dev_p;
    float* dev_r;
    float* dev_p2;
    float* dev_r2;
    int error = 0;
    // 创建主机上的页锁定内存 用于异步的和GPU之间进行数据拷贝
    error = cudaHostAlloc((void**)&host_p, N * DIM * DIM * sizeof(float), cudaHostAllocDefault);
    // printf("Error code cudaHostAlloc: %d\n", error);
    for (int k = 0; k < N; k++)
    {
        float* tmp = host_p + k * DIM * DIM;
        for (int i = 0; i < DIM; i++)
        {
            for (int j =0; j < DIM; j++)
            {
                *(tmp + i * DIM + j) = 100.0 * k;
            }
        }
    }
    error = cudaHostAlloc((void**)&host_r, N * OUTPUT_DIM * OUTPUT_DIM * sizeof(float), cudaHostAllocDefault);
    error = cudaMalloc((void**)&dev_p, DIM * DIM * sizeof(float));
    error = cudaMalloc((void**)&dev_r, OUTPUT_DIM * OUTPUT_DIM * sizeof(float));
    error = cudaMalloc((void**)&dev_p2, DIM * DIM * sizeof(float));
    error = cudaMalloc((void**)&dev_r2, OUTPUT_DIM * OUTPUT_DIM * sizeof(float));

    // 创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);

    // 创建cuda事件 用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 迭代计算N个输入
    dim3 threads(OUTPUT_DIM, OUTPUT_DIM);
    for (int i = 0; i < N; i += 2)
    {
        // 在向流中传递操作时 遵循宽度优先的方式 这样能够避免底层硬件进行调度时带来的效率低下问题
        cudaMemcpyAsync(dev_p, host_p + i * DIM * DIM, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_p2, host_p + (i + 1) * DIM * DIM, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice, stream2);
        AveragePooling2D<<<1, threads, 0, stream>>>(dev_p, dev_r);
        AveragePooling2D<<<1, threads, 0, stream2>>>(dev_p2, dev_r2);
        error = cudaMemcpyAsync(host_r + i * OUTPUT_DIM * OUTPUT_DIM, dev_r, OUTPUT_DIM * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream);
        error = cudaMemcpyAsync(host_r + (i + 1) * OUTPUT_DIM * OUTPUT_DIM, dev_r2, OUTPUT_DIM * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream2);
        
        // cudaMemcpyAsync(dev_p, host_p + i * DIM * DIM, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice, stream);
        // AveragePooling2D<<<1, threads, 0, stream>>>(dev_p, dev_r);
        // error = cudaMemcpyAsync(host_r + i * OUTPUT_DIM * OUTPUT_DIM, dev_r, OUTPUT_DIM * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream);
        // cudaMemcpyAsync(dev_p2, host_p + (i + 1) * DIM * DIM, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice, stream2);
        // AveragePooling2D<<<1, threads, 0, stream2>>>(dev_p2, dev_r2);
        // error = cudaMemcpyAsync(host_r + (i + 1) * OUTPUT_DIM * OUTPUT_DIM, dev_r2, OUTPUT_DIM * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    }
    cudaStreamSynchronize(stream);
    cudaStreamSynchronize(stream2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time Cost:  %3.1f ms\n", elapsedTime );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(stream2);
    cudaFree(dev_p);
    cudaFree(dev_r);
    cudaFree(dev_p2);
    cudaFree(dev_r2);
    // for (int k = 0; k < N; k++)
    // {
    //     printMatrix(host_r + k * OUTPUT_DIM * OUTPUT_DIM, OUTPUT_DIM, OUTPUT_DIM);
    // }
    cudaFreeHost(host_p);
    cudaFreeHost(host_r);
    return 0;
}

