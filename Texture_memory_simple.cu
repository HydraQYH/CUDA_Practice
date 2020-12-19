#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#define N 10

texture<float> texOriginal;

__global__ void copyData(float* p_destination)
{
    *(p_destination + threadIdx.x) = tex1Dfetch(texOriginal, threadIdx.x);
}

void printArray(float* p, int len)
{
    printf("{ ");
    for (int i = 0; i < len; i++)
    {
        if (i != len - 1)
        {
            printf("%.3f, ", *(p + i));
        }
        else
        {
            printf("%.3f }\n", *(p + i));
        }
    }
}

int main(void)
{
    // 在CPU上创建数组
    float a[N];
    for (int i = 0; i < N; i++)
    {
        a[i] = 1.0;
    }
    float b[N];
    for (int i = 0; i < N; i++)
    {
        b[i] = 0.0;
    }
    printArray(a, N);
    printArray(b, N);
    // 在GPU上为源数组和目的数组分配空间
    float* a_dev;
    float* b_dev;
    cudaMalloc((void**)&a_dev, N * sizeof(float));
    cudaMalloc((void**)&b_dev, N * sizeof(float));
    // 绑定纹理内存
    int error_code = cudaBindTexture(NULL, texOriginal, a_dev, N * sizeof(float));
    printf("cudaBindTexture return code: %d\n", error_code);

    // 将源数据拷贝至GPU
    cudaMemcpy(a_dev, a, N * sizeof(float), cudaMemcpyHostToDevice);
    // 在GPU上进行拷贝
    copyData<<<1, N>>>(b_dev);
    // 将GPU 上的目的数组拷贝回CPU
    cudaMemcpy(b, b_dev, N * sizeof(float), cudaMemcpyDeviceToHost);
    // 任务结束时解除纹理内存的绑定
    cudaUnbindTexture(texOriginal);
    cudaFree(a_dev);
    cudaFree(b_dev);
    printArray(a, N);
    printArray(b, N);
    return 0;
}

