#include <stdio.h>
#include "book.h"
#define M 5
#define N 4
#define K 3
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
    int result = 0;
    int offset_a = blockIdx.x * d;  // matrix_a row blockIdx.x
    int offset_b = blockIdx.y;          // matrix_b column blockIdx.y
    for (int i = 0; i < d; i++)
    {
        // matrix_a[blockIdx.x][i] * matrix[i][blockIdx.y]
        result += pa[offset_a] * pb[offset_b];
        offset_a++;
        offset_b += gridDim.y;
    }
    int offset_c = blockIdx.x * gridDim.y + blockIdx.y;
    *(pc + offset_c) = result;
}

__global__ void matmul2(int* pa, int * pb, int* pc, int d)
{
    // 使用线程块内的共享内存
    __shared__ int buffer[MAX_BUFFER_SIZE];
    int buffer_size = (d + 1) / 2 * 2;

    int result = 0;
    int offset_a = blockIdx.x * d;  // matrix_a row blockIdx.x
    int offset_b = blockIdx.y;          // matrix_b column blockIdx.y
    int offset_d = threadIdx.x;

    offset_a += offset_d;
    offset_b += (offset_d * gridDim.y);
    result = pa[offset_a] * pb[offset_b];
    buffer[offset_d] = result;
    
    if (offset_d == 0 && (d & 0x1) == 1)
    {
        // 当d为奇数时 迫使0号进程将buffer末尾处填充0
        buffer[buffer_size - 1] = 0;
    }
    __syncthreads();
    result = 0;
    int i = buffer_size / 2;
    while (i != 0)
    {
        if (offset_d < i)
        {
            buffer[offset_d] += buffer[offset_d + i];
        }
        __syncthreads();
        i /= 2;
    }
    // 最终和值在buffer[0]处
    *(pc + blockIdx.x * gridDim.y + blockIdx.y) = buffer[0];
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

    dim3 grid(M, K);
    matmul2<<<grid, N>>>(dev_a, dev_b, dev_c, N);
    // matmul<<<grid, 1>>>(dev_a, dev_b, dev_c, N);
    
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

