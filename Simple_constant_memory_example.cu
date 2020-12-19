#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#define M 3
#define N 4
#define T 32

__constant__ float origin_matrix[M][N];

__global__ void transposition(float* p_res_dev)
{
    /**
     * @brief 
     * 所有的线程读取的数据都是完全相同的 这时使用常量内存可以提升程序性能
     * 原因有2点：
     *      1. 每个半线程束（16线程）在读取数据时会进行广播，原本需要16个线程同时进行16个内存请求，
     *      现在就只需要一个内存请求，减少了内存带宽占用率。
     *      2. 当有任意一个线程（半线程束）进行了内存请求后，请求的内容会存放到缓存中，其他的半线程束
     *      再次请求同样的内存时，就可以直接从缓存中读取。
     */
    float result = 0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result += origin_matrix[i][j];
        }
    }
    *(p_res_dev + threadIdx.x) = result * (threadIdx.x + 1);
}

int main(void)
{
    float data[M][N];
    float* p_res_dev = NULL;
    float res[T];
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            data[i][j] = i * N + j;
        }
    }
    cudaMemcpyToSymbol(origin_matrix, data, sizeof(float) * M * N);
    cudaMalloc((void**)&p_res_dev, sizeof(float) * T);
    transposition<<<1, T>>>(p_res_dev);
    cudaMemcpy(res, p_res_dev, sizeof(float) * T, cudaMemcpyDeviceToHost);
    for (int i = 0; i < T; i++)
    {
        printf("%d: %d\n", i, int(res[i]));
    }
    cudaFree(p_res_dev);
    return 0;
}

