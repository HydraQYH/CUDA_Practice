#include <stdio.h>
#include <stdlib.h>
#define M 32
#define D 8
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
    // 创建进程块内部共享内存缓冲区
    __shared__ int buffer[16][16][D];
    // 线程块坐标(blockIdx.x, blockIdx.y)
    // 线程坐标(threadIdx.x, threadIdx.y, threadIdx.z)
    int offset_row = blockIdx.x * blockDim.x + threadIdx.x;
    int offset_column = blockIdx.y * blockDim.y + threadIdx.y;
    int tmp1 = *(p_a + offset_row * D + threadIdx.z);
    int tmp2 = *(p_b + offset_column + threadIdx.z * N);
    buffer[threadIdx.x][threadIdx.y][threadIdx.z] = tmp1 * tmp2;
    __syncthreads();
    int i = D / 2;
    while (i != 0)
    {
        if (threadIdx.z < i)
        {
            buffer[threadIdx.x][threadIdx.y][threadIdx.z] += buffer[threadIdx.x][threadIdx.y][threadIdx.z + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.z == 0)
    {
        *(p_c + offset_row * N + offset_column) = buffer[threadIdx.x][threadIdx.y][0];
    }
}

int main(void)
{
    int a[M][D];
    int b[D][N];
    int c[M][N];
    int* dev_a;
    int* dev_b;
    int* dev_c;

    cudaMalloc((void**)&dev_a, M * D * sizeof(int));
    cudaMalloc((void**)&dev_b, D * N * sizeof(int));
    cudaMalloc((void**)&dev_c, M * N * sizeof(int));

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < D; j++)
        {
            a[i][j] = 2;
        }
    }

    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < N; j++)
        {
            b[i][j] = 2;
        }
    }

    printf("Matrix A:\n");
    printMatrix((int*)a, M, D);
    printf("Matrix B:\n");
    printMatrix((int*)b, D, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpy(dev_a, a, M * D *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, D * N *sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 grid(M / 16, N / 16);
    dim3 block(16, 16, D);
    matmul_advance<<<grid, block>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, M * N *sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);

    printf("Matrix C:\n");
    printMatrix((int*)c, M, N);

    return 0;
}

