#include <stdio.h>
#include <malloc.h>
#include "cuda.h"
const int B = 8;
const int S = 6;
const int H = 3;

__global__ void transform201(float* p_source, float* p_destination)
{
    int tid = threadIdx.x;
    int h = blockDim.x;
    int bid_x = blockIdx.x;
    int b = gridDim.x;
    int bid_y = blockIdx.y;
    int s = gridDim.y;

    int input_stride_0 = s * h;
    int input_stride_1 = h;
    int output_stride_0 = b * s;
    int output_stride_1 = s;

    float val = p_source[input_stride_0 * bid_x + input_stride_1 * bid_y + tid];
    p_destination[output_stride_0 * tid + output_stride_1 * bid_x + bid_y] = val * val;
}

void _init_source_content(float* p)
{
    float count = 0;
    for (int i = 0; i < B; i++)
    {
        for (int j = 0; j < S; j++)
        {
            for (int k = 0; k < H; k++)
            {
                p[i * S * H + j * H + k] = count;
                count += 1;
            }
        }
    }
}

void printArray(float* p)
{
    for (int i = 0; i < B * S * H; i++)
    {
        printf("%.1f ", p[i]);
    }
    printf("\n");
}

void CPUTransform(float* p_s, float* p_d)
{
    for (int i = 0; i < B; i++)
    {
        for (int j = 0; j < S; j++)
        {
            for (int k = 0; k < H; k++)
            {
                float val = p_s[i * S * H + j * H + k];
                p_d[k * B * S + i * S + j] = val * val;
            }
        }
    }
}

int main(void)
{
    /*
    * Shape Transforme: (B, S, H) -> (H, B, S)  (8, 6, 3) -> (3, 8, 6)
    * grid: B * S block: H 
    */
    float* h_s = NULL;
    float* h_d = NULL;
    float* d_s = NULL;
    float* d_d = NULL;
    // 在主机上分配内存
    h_s = (float*)malloc(sizeof(float) * B * S * H);
    // 初始化源内容
    _init_source_content(h_s);
    h_d = (float*)malloc(sizeof(float) * H * B * S);
    printArray(h_s);
    // 在设备上分配内存
    cudaMalloc((void**)&d_s, sizeof(float) * B * S * H);
    cudaMalloc((void**)&d_d, sizeof(float) * H * B * S);
    // 拷贝源数据
    cudaMemcpy(d_s, h_s, sizeof(float) * B * S * H, cudaMemcpyHostToDevice);
    // 定义grid block
    dim3 grid(B, S);
    dim3 block(H);
    transform201<<<grid, block>>>(d_s, d_d);
    // 取回数据
    cudaMemcpy(h_d, d_d, sizeof(float) * H * B * S, cudaMemcpyDeviceToHost);
    printArray(h_d);
    // 验证转换是否正确
    CPUTransform(h_s, h_d);
    printArray(h_d);
    cudaFree(d_d);
    cudaFree(d_s);
    free(h_d);
    free(h_s);
    return 0;
}

