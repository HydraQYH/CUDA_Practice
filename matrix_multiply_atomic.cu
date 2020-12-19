#include <stdio.h>
#include "book.h"
#define M 5
#define N 4
#define K 4
#define MAX_BUFFER_SIZE 1024

void printMatrix(int* p, int row, int column)
{
    for (int i = 0; i < row; i++)
    {
        printf("[ ");
        for (int j = 0; j < column; j++)
        {
            if (j != column - 1)
            {
                printf("%d, ", *(p + i * column + j));
            }
            else
            {
                printf("%d ]\n", *(p + i * column + j));
            }
        }
    }
}

__global__ void matmul(int* pa, int * pb, int* pc, int d)
{
    int offset_a = blockIdx.x * d +  threadIdx.x;
    int offset_b = blockIdx.y +  threadIdx.x * gridDim.y;
    int offset_c = blockIdx.x * gridDim.y + blockIdx.y;
    // 直接在全局内存上做原子操作
    // 这样效率未必会很高 相当于N个线程在竞争内存上的一个位置 当N很大时  效率会低下
    atomicAdd(pc + offset_c, pa[offset_a] * pb[offset_b]);
}

__global__ void matmul2(int* pa, int * pb, int* pc, int d)
{
    int res = 0;
    __shared__ int buffer;
    if (threadIdx.x == 0)
    {
        buffer = 0;
    }
    __syncthreads();
    int offset_a = blockIdx.x * d +  threadIdx.x;
    int offset_a_ = offset_a + d / 2;
    int offset_b = blockIdx.y +  threadIdx.x * gridDim.y;
    int offset_b_ = offset_b + (d / 2) *  gridDim.y;
    int offset_c = blockIdx.x * gridDim.y + blockIdx.y;
    // 使用共享内存原子操作 此时也只有N / 2个线程竞争一个内存地址 提升效率
    res = pa[offset_a] * pb[offset_b] + pa[offset_a_] * pb[offset_b_];
    atomicAdd(&buffer, res);
    if (threadIdx.x == 0)
    {
        *(pc + offset_c) = buffer;
    }
}


int main()
{
    int matrix_a[M][N];
    int matrix_b[N][K];
    int matrix_c[M][K];
    int* dev_a;
    int* dev_b;
    int* dev_c;

    int tmp = 1;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix_a[i][j] = 1;
            tmp++;
        }
    }
    printf("Matrix A:\n");
    printMatrix((int*)matrix_a, M, N);

    tmp = 10;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            matrix_b[i][j] = 1;
            tmp--;
        }
    }
    printf("Matrix B:\n");
    printMatrix((int*)matrix_b, N, K);

    cudaMalloc((void**)&dev_a, M * N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * K * sizeof(int));
    cudaMalloc((void**)&dev_c, M * K * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemcpy(dev_a, matrix_a, M * N *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, matrix_b, N * K *sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemset(dev_c, 0, M * K * sizeof(int));  // 先将输出矩阵所有元素置0 为后续累加做准备

    dim3 grid(M, K);
    // matmul<<<grid, N>>>(dev_a, dev_b, dev_c, N);
    matmul2<<<grid, N / 2>>>(dev_a, dev_b, dev_c, N);
    
    cudaMemcpy(matrix_c, dev_c, M * K *sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_b);
    printf("Matrix C:\n");
    printMatrix((int*)matrix_c, M, K);
    return 0;
}

