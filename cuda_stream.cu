#include <stdio.h>
#include <stdlib.h>
#define M 32
#define D 3
#define N 32

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

__global__ void matmul_advance(int* p_a, int* p_b, int* p_c)
{
    // 创建线程块内部共享内存缓冲区
    __shared__ int buffer[16][16][D];
    // 线程块坐标(blockIdx.x, blockIdx.y)
    // 线程坐标(threadIdx.x, threadIdx.y, threadIdx.z)
    int offset_row = blockIdx.x * blockDim.x + threadIdx.x;
    int offset_column = blockIdx.y * blockDim.y + threadIdx.y;
    int tmp1 = *(p_a + offset_row * D + threadIdx.z);
    int tmp2 = *(p_b + offset_column + threadIdx.z * N);
    buffer[threadIdx.x][threadIdx.y][threadIdx.z] = tmp1 * tmp2;
    __syncthreads();
    int i = D;
    while (i != 1)
    {
        int half = i / 2;
        int odd = i % 2;
        if (threadIdx.z < half)
        {
            buffer[threadIdx.x][threadIdx.y][threadIdx.z] += buffer[threadIdx.x][threadIdx.y][threadIdx.z + half];
        }
        if (odd == 1 && threadIdx.z == 0)
        {
            buffer[threadIdx.x][threadIdx.y][0] += buffer[threadIdx.x][threadIdx.y][half];
        }
        __syncthreads();
        i = half;
    }
    if (threadIdx.z == 0)
    {
        *(p_c + offset_row * N + offset_column) = buffer[threadIdx.x][threadIdx.y][0];
    }
}

int main(void)
{
    int* host_a;
    int* host_b;
    int* host_c;

    int* dev_a;
    int* dev_b;
    int* dev_c;

    // 主机上存储原始矩阵的内存分配为页锁定内存
    cudaHostAlloc((void**)&host_a, M * D * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, D *  N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, M * N * sizeof(int), cudaHostAllocDefault);

    cudaMalloc((void**)&dev_a, M * D * sizeof(int));
    cudaMalloc((void**)&dev_b, D * N * sizeof(int));
    cudaMalloc((void**)&dev_c, M * N * sizeof(int));

    // 初始化两个输入矩阵
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < D; j++)
        {
            *(host_a + i * D + j) = 1;
        }
    }

    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < N; j++)
        {
            *(host_b + i * N + j) = 1;
        }
    }
    printf("Matrix A:\n");
    printMatrix(host_a, M, D);
    printf("Matrix B:\n");
    printMatrix(host_b, D, N);

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 初始化并启动计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dim3 grid(M / 16, N / 16);
    // D的最大值为4
    dim3 block(16, 16, D);

    // 将两个输入矩阵copy到GPU上(以异步的方式)
    cudaMemcpyAsync(dev_a, host_a, M * D * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_b, host_b, D * N * sizeof(int), cudaMemcpyHostToDevice, stream);
    
    // 做矩阵乘法
    matmul_advance<<<grid, block, 0, stream>>>(dev_a, dev_b, dev_c);
    // 将计算结果拷贝回主机页锁定内存
    cudaMemcpyAsync(host_c, dev_c, M * N *sizeof(int), cudaMemcpyDeviceToHost, stream);
    // 计算运行时间
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );
    
    printf("Matrix C:\n");
    printMatrix(host_c, M, N);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);

    return 0;
}

