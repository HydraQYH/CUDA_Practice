#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#define N 5

texture<float, 2> texOriginal;

__global__ void copyData(float* p_destination)
{
    int offset = threadIdx.y * blockDim.x + threadIdx.x;
    *(p_destination + offset) = tex2D(texOriginal, threadIdx.y, threadIdx.x);
}

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

int main(void)
{
    // 在CPU上创建数组
    float a[N][N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i][j] = 1.0;
        }
    }
    float b[N][N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            b[i][j] = 0.0;
        }
    }
    printMatrix((float*)a, N, N);
    printMatrix((float*)b, N, N);
    // 在GPU上为源数组和目的数组分配空间
    float* a_dev;
    float* b_dev;

    size_t a_pitch;
    cudaMallocPitch((void**)&a_dev, &a_pitch, N * sizeof(float), N);
    cudaMalloc((void**)&b_dev, N * N * sizeof(float));
    printf("Original Matrix pithch: %lu\n", a_pitch);
    // 绑定纹理内存
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    /**
     * @brief 
     * cudaErrorInvalidValue = 11
     * This indicates that one or more of the parameters
     * passed to the API call is not within an acceptable range of values.
     */
    int error_code = cudaBindTexture2D(0, texOriginal, a_dev, desc, N, N, a_pitch);
    printf("cudaBindTexture2D return code: %d\n", error_code);

    // 将源数据拷贝至GPU
    // 当源数据在CPU内存上时 其pitch为二维矩阵每一行所占的字节数
    cudaMemcpy2D(a_dev, a_pitch, a, N * sizeof(float), N * sizeof(float), N, cudaMemcpyHostToDevice);
    // 在GPU上进行拷贝
    dim3 threads(N, N);
    copyData<<<1, threads>>>(b_dev);
    // 将GPU 上的目的数组拷贝回CPU
    cudaMemcpy(b, b_dev, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaUnbindTexture(texOriginal);
    cudaFree(a_dev);
    cudaFree(b_dev);
    printMatrix((float*)a, N, N);
    printMatrix((float*)b, N, N);
    return 0;
}

