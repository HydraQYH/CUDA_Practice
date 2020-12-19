#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#define N 10
#define DIM 5
#define OUTPUT_DIM 3

texture<float, 2> tex;

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

__global__ void AveragePooling2D(float* p_result)
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
            result += tex2D(tex, x + i, y + j);
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
    int error = 0;
    // 创建主机上的页锁定内存 用于异步的和GPU之间进行数据拷贝
    error = cudaHostAlloc((void**)&host_p, N * DIM * DIM * sizeof(float), cudaHostAllocDefault);
    printf("Error code cudaHostAlloc: %d\n", error);
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
    for (int k = 0; k < N; k++)
    {
        printMatrix(host_p + k * DIM * DIM, DIM, DIM);
    }
    error = cudaHostAlloc((void**)&host_r, N * OUTPUT_DIM * OUTPUT_DIM * sizeof(float), cudaHostAllocDefault);
    printf("Error code cudaHostAlloc: %d\n", error);
    // 创建GPU上的缓存空间
    size_t pitch;
    error = cudaMallocPitch((void**)&dev_p, &pitch, DIM * sizeof(float), DIM);
    printf("Error code cudaMallocPitch: %d\n", error);
    error = cudaMalloc((void**)&dev_r, OUTPUT_DIM * OUTPUT_DIM * sizeof(float));
    printf("Error code cudaMalloc: %d\n", error);
    // 对于输入 将其绑定到二维纹理内存上
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    error = cudaBindTexture2D(NULL, tex, dev_p, desc, DIM, DIM, pitch);
    printf("Error code cudaBindTexture2D: %d\n", error);
    // 创建cuda流
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);

    // 创建cuda事件 用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 迭代计算N个输入
    for (int i = 0; i < N; i++)
    {
        // 通过同步方式将输入数据copy至GPU
        error = cudaMemcpy2D(dev_p, pitch, host_p + i * DIM * DIM, DIM * sizeof(float), DIM * sizeof(float), DIM, cudaMemcpyHostToDevice);
        printf("Error code cudaMemcpy2D: %d\n", error);
        // 调用核函数
        dim3 threads(OUTPUT_DIM, OUTPUT_DIM);
        AveragePooling2D<<<1, threads>>>(dev_r);
        // 通过异步方式将输出数据copy至CPU
        error = cudaMemcpy(host_r + i * OUTPUT_DIM * OUTPUT_DIM, dev_r, OUTPUT_DIM * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Error code cudaMemcpy: %d\n", error);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time Cost:  %3.1f ms\n", elapsedTime );
    cudaUnbindTexture(tex);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // cudaStreamDestroy(stream);
    cudaFree(dev_p);
    cudaFree(dev_r);
    for (int k = 0; k < N; k++)
    {
        printMatrix(host_r + k * OUTPUT_DIM * OUTPUT_DIM, OUTPUT_DIM, OUTPUT_DIM);
    }
    cudaFreeHost(host_p);
    cudaFreeHost(host_r);
    return 0;
}

