#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
const int TILE_DIM = 32;
const int M = 128 * TILE_DIM;

__global__ void transpose_matrix_multiply_baseline(float* p_a, float* p_c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    // 目标矩阵的row行col列 应该是矩阵A的第row行和第col行的dot product
    for (int i = 0; i < TILE_DIM; i++)
    {
        sum += p_a[row * TILE_DIM + i] * p_a[col * TILE_DIM + i];
    }
    p_c[row * M + col] = sum;
}

__global__ void transpose_matrix_multiply_1(float* p_a, float* p_c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    __shared__ float a_tile[TILE_DIM][TILE_DIM];
    __shared__ float a_t_tile[TILE_DIM][TILE_DIM];
    a_tile[threadIdx.y][threadIdx.x] = p_a[row * TILE_DIM + threadIdx.x];
    a_t_tile[threadIdx.x][threadIdx.y] = p_a[blockIdx.x * blockDim.x * TILE_DIM + threadIdx.y * TILE_DIM + threadIdx.x];
    for (int i = 0; i < TILE_DIM; i++)
    {
        sum += a_tile[threadIdx.y][i] * a_t_tile[i][threadIdx.x];
    }
    p_c[row * M + col] = sum;
}

__global__ void transpose_matrix_multiply_2(float* p_a, float* p_c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    __shared__ float a_tile[TILE_DIM][TILE_DIM];
    __shared__ float a_t_tile[TILE_DIM][TILE_DIM + 1];
    a_tile[threadIdx.y][threadIdx.x] = p_a[row * TILE_DIM + threadIdx.x];
    a_t_tile[threadIdx.x][threadIdx.y] = p_a[blockIdx.x * blockDim.x * TILE_DIM + threadIdx.y * TILE_DIM + threadIdx.x];
    for (int i = 0; i < TILE_DIM; i++)
    {
        sum += a_tile[threadIdx.y][i] * a_t_tile[i][threadIdx.x];
    }
    p_c[row * M + col] = sum;
}

int main(void)
{
    float* d_a = nullptr;
    float* d_c = nullptr;
    cudaMalloc((void**)&d_a, M * TILE_DIM * TILE_DIM * sizeof(float));
    cudaMalloc((void**)&d_c, M * M * sizeof(float));
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 grid(M / TILE_DIM, M / TILE_DIM);
    dim3 block(TILE_DIM, TILE_DIM);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
    {
        // transpose_matrix_multiply_baseline<<<grid, block>>>(d_a, d_c);
        // transpose_matrix_multiply_1<<<grid, block>>>(d_a, d_c);
        transpose_matrix_multiply_2<<<grid, block>>>(d_a, d_c);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    std::cout << "Elapsed Time: " << elapsed_time << " ms" << std::endl;

    cudaEventDestroy(end);
    cudaEventDestroy(start);
    
    cudaFree(d_a);
    cudaFree(d_c);
}

