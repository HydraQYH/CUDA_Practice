#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#define FEATURE_DIM 38
#define FILTER_DIM 7
#define STRIDE 1
#define PADDING 0

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

__global__ void conv(float* p_input, float* p_filter, float* p_output)
{
    // 当前线程要计算输出中(threadIdx.x, threadIdx.y)的值
    int output_offset = threadIdx.y + threadIdx.x * blockDim.y;
    // 获取当前输出位置所对应的输入区域中心
    int input_x_offset = FILTER_DIM/ 2 + STRIDE * threadIdx.x;
    int input_y_offset = FILTER_DIM / 2 + STRIDE * threadIdx.y;
    // 区域左上角
    int input_x_offset_left_up = input_x_offset - FILTER_DIM / 2;
    int input_y_offset_left_up = input_y_offset - FILTER_DIM / 2;

    float result = 0;
    for (int i = 0; i < FILTER_DIM; i++)
    {
        for (int j = 0; j < FILTER_DIM; j++)
        {
            int x = input_x_offset_left_up + i;
            int y = input_y_offset_left_up + j;
            int offset = x * (FEATURE_DIM + 2 * PADDING) + y;
            result += p_input[offset] * p_filter[i * FILTER_DIM + j];
        }
    }
    p_output[output_offset] = result;
}

int main(void)
{
    // 创建输入图像
    float input[FEATURE_DIM + 2 * PADDING][FEATURE_DIM + 2 * PADDING];
    for (int i = 0; i < FEATURE_DIM + 2 * PADDING; i++)
    {
        for (int j = 0; j < FEATURE_DIM + 2 * PADDING; j++)
        {
            // input[i][j] = i * (FEATURE_DIM + 2 * PADDING) + 1 + j;
            input[i][j] = 1;
        }
    }
    printf("Input Feature Map:\n");
    printMatrix((float*)input, FEATURE_DIM + 2 * PADDING, FEATURE_DIM + 2 * PADDING);
    // 创建卷积核
    float filter[FILTER_DIM][FILTER_DIM];
    for (int i = 0; i < FILTER_DIM; i++)
    {
        for (int j = 0; j < FILTER_DIM; j++)
        {
            // filter[i][j] = i * FILTER_DIM + 1 + j;
            filter[i][j] = 1;
        }
    }
    printf("Filter:\n");
    printMatrix((float*)filter, FILTER_DIM, FILTER_DIM);
    // 创建输出
    int output_dim = (FEATURE_DIM + 2 *PADDING - FILTER_DIM) / STRIDE + 1;
    float* p_output = (float*)malloc(output_dim * output_dim * sizeof(float));

    float* input_dev;
    float* filter_dev;
    float* output_dev;

    // 在GPU上为输入输出以及卷积核分配空间
    cudaMalloc((void**)&input_dev, (FEATURE_DIM + 2 * PADDING) * (FEATURE_DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc((void**)&filter_dev, (FILTER_DIM) * (FILTER_DIM) * sizeof(float));
    cudaMalloc((void**)&output_dev, output_dim * output_dim * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpy(input_dev, input, (FEATURE_DIM + 2 * PADDING) * (FEATURE_DIM + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(filter_dev, filter, FILTER_DIM * FILTER_DIM * sizeof(float), cudaMemcpyHostToDevice);
    // 只使用一个线程块
    dim3 threads(output_dim, output_dim);
    conv<<<1, threads>>>(input_dev, filter_dev, output_dev);
    cudaMemcpy(p_output, output_dev, output_dim * output_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cudaFree(input_dev);
    cudaFree(filter_dev);
    cudaFree(output_dev);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Output Feature map:\n");
    printMatrix((float*)p_output, output_dim, output_dim);
    free(p_output);
    printf( "Time Cost:  %3.1f ms\n", elapsedTime );
    return 0;
}

