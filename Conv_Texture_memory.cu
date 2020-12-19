#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#define FEATURE_DIM 5
#define FILTER_DIM 3
#define STRIDE 1
#define PADDING 0

// 将纹理内存引用声明为全局变量
texture<float, 2> texInput;
texture<float, 2> texFilter;

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

__global__ void conv(float* p_output)
{
    // threadIdx.x负责width threadIdx.y负责heigth
    int output_offset = threadIdx.x + threadIdx.y * blockDim.x;
    // 获取当前输出位置所对应的输入区域中心
    int input_w_offset = FILTER_DIM / 2 + STRIDE * threadIdx.x;
    int input_h_offset = FILTER_DIM / 2 + STRIDE * threadIdx.y;
    int filter_w_offset = FILTER_DIM / 2;
    int filter_h_offset = FILTER_DIM / 2;

    float result = 0;
    for (int i = -(FILTER_DIM / 2); i <= FILTER_DIM / 2; i++)
    {
        for (int j = -(FILTER_DIM / 2); j <= FILTER_DIM / 2; j++)
        {
            float a = tex2D(texInput, input_w_offset + i, input_h_offset + j);
            float b = tex2D(texFilter, filter_w_offset + i, filter_h_offset + j);
            result += a * b;
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
            input[i][j] = i + 1;
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
    size_t input_dev_pitch;
    cudaMallocPitch((void**)&input_dev, &input_dev_pitch,
    (FEATURE_DIM + 2 * PADDING) * sizeof(float), (FEATURE_DIM + 2 * PADDING));
    size_t filter_dev_pitch;
    cudaMallocPitch((void**)&filter_dev, &filter_dev_pitch, FEATURE_DIM * sizeof(float), FILTER_DIM);
    cudaMalloc((void**)&output_dev, output_dim * output_dim * sizeof(float));
    // 绑定二维纹理内存
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, texInput, input_dev, desc,
        (FEATURE_DIM + 2 * PADDING),
        (FEATURE_DIM + 2 * PADDING),
        input_dev_pitch);
    cudaBindTexture2D(NULL, texFilter, filter_dev, desc,
        FILTER_DIM, FILTER_DIM, filter_dev_pitch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpy2D(input_dev, input_dev_pitch, input,
        (FEATURE_DIM + 2 * PADDING) * sizeof(float),
        (FEATURE_DIM + 2 * PADDING) * sizeof(float),
        FEATURE_DIM + 2 * PADDING, cudaMemcpyHostToDevice);
    cudaMemcpy2D(filter_dev, filter_dev_pitch, filter,
        FILTER_DIM * sizeof(float),
        FILTER_DIM * sizeof(float),
        FILTER_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_dev, filter, FILTER_DIM * FILTER_DIM * sizeof(float), cudaMemcpyHostToDevice);
    // 只使用一个线程块
    dim3 threads(output_dim, output_dim);
    conv<<<1, threads>>>(output_dev);
    cudaMemcpy(p_output, output_dev, output_dim * output_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cudaUnbindTexture(texInput);
    cudaUnbindTexture(texFilter);
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

