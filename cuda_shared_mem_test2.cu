#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
const int TILE_DIM = 32;
const int M = 128 * TILE_DIM;
const int N = 64 * TILE_DIM;


__global__ void matrix_multiply_baseline(float* p_a, float* p_b, float* p_c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    // 一个线程束内的线程取矩阵A同一行的值
    for (int i = 0; i < TILE_DIM; i++)
    {
        // 每次迭代 同一线程束内相邻线程取矩阵B相邻的元素
        sum += p_a[row * TILE_DIM + i] * p_b[i * N + col];
    }
    p_c[row * N + col] = sum;
}

__global__ void matrix_multiply_shared_memory_1(float* p_a, float* p_b, float* p_c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tile_a[TILE_DIM][TILE_DIM];
    tile_a[threadIdx.y][threadIdx.x] = p_a[row * TILE_DIM + threadIdx.x];
    float sum = 0.0;
    for (int i = 0; i < TILE_DIM; i++)
    {
        // 同一线程束内相邻线程访问相同元素 共享内存可以广播
        sum += tile_a[threadIdx.y][i] * p_b[i * N + col];
    }
    p_c[row * N + col] = sum;
}

__global__ void matrix_multiply_shared_memory_2(float* p_a, float* p_b, float* p_c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tile_a[TILE_DIM][TILE_DIM];
    __shared__ float tile_b[TILE_DIM][TILE_DIM];
    tile_a[threadIdx.y][threadIdx.x] = p_a[row * TILE_DIM + threadIdx.x];
    tile_b[threadIdx.y][threadIdx.x] = p_b[threadIdx.y * N + col];
    float sum = 0.0;
    for (int i = 0; i < TILE_DIM; i++)
    {
        // 对于矩阵B 同一线程束内相邻线程访问相邻元素 无bank冲突
        sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
    }
    p_c[row * N + col] = sum;
}

int main(void)
{
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    cudaMalloc((void**)&d_a, M * TILE_DIM * TILE_DIM * sizeof(float));
    cudaMalloc((void**)&d_b, TILE_DIM * N * TILE_DIM * sizeof(float));
    cudaMalloc((void**)&d_c, M * N * sizeof(float));
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 grid(N / TILE_DIM, M / TILE_DIM);
    dim3 block(TILE_DIM, TILE_DIM);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
    {
        // matrix_multiply_baseline<<<grid, block>>>(d_a, d_b, d_c);
        // matrix_multiply_shared_memory_1<<<grid, block>>>(d_a, d_b, d_c);
        matrix_multiply_shared_memory_2<<<grid, block>>>(d_a, d_b, d_c);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    std::cout << "Elapsed Time: " << elapsed_time << " ms" << std::endl;

    cudaEventDestroy(end);
    cudaEventDestroy(start);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

