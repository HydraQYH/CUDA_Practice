#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
const int B = 4;
const int S = 8;
const int C = 3;
const int A = 12;
const int N = 16;    // 假定可以被4整除

__global__ void transform(float4* p_source, float4* p_dst)
{
    int b = blockIdx.x;
    int s = blockIdx.y;
    int c = blockIdx.z;
    int a = threadIdx.x;
    int n_f4 = threadIdx.y;

    int _B = gridDim.x;
    int _S = gridDim.y;
    int _C = gridDim.z;
    int _A = blockDim.x;
    int _N_f4 = blockDim.y;

    int input_stride_3 = _N_f4;
    int input_stride_2 = _A * input_stride_3;
    int input_stride_1 = _C * input_stride_2;
    int input_stride_0 = _S * input_stride_1;

    int output_stride_3 = _N_f4;
    int output_stride_2 = _S * output_stride_3;
    int output_stride_1 = _A * output_stride_2;
    int output_stride_0 = _B * output_stride_1;
    
    // 每个线程处理一个float4类型元素
    int source_index = b * input_stride_0 + s * input_stride_1 + c * input_stride_2 + a * input_stride_3 + n_f4;
    int dst_index = c * output_stride_0 + b * output_stride_1 + a * output_stride_2 + s * output_stride_3 + n_f4;
    p_dst[dst_index] = p_source[source_index];
}

void _init_host_buffer(float* p)
{
    for (int b = 0; b < B; b++)
    {
        for (int s = 0; s < S; s++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int a = 0; a < A; a++)
                {
                    for (int n = 0; n < N; n++)
                    {
                        p[b * S * C * A * N + s * C * A * N + c * A * N + a * N + n] = 1.0 * ( rand() % RAND_MAX ) / RAND_MAX;       
                    }   
                }
            }
        }
    }
}

void cpu_transform(float* source, float* d)
{
    for (int b = 0; b < B; b++)
    {
        for (int s = 0; s < S; s++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int a = 0; a < A; a++)
                {
                    for (int n = 0; n < N; n++)
                    {
                        d[c * B * A * S * N + b * A * S * N + a * S * N + s * N + n] = source[b * S * C * A * N + s * C * A * N + c * A * N + a * N + n];   
                    }   
                }
            }
        }
    }
}

void print(float* p)
{
    for (int i = 0; i < C * B * A * S * N; i++)
    {
        printf("%.5f ", p[i]);
    }
    printf("\n");
}

int main()
{
    // {B S C H} H == A * N -> {C, B, A, S, N}
    // {B S C A N} -> {C B A S N}
    // 计算以float4为单位的 输入最后一维元素总数
    int N_f4 = N / 4;
    dim3 grid(B, S, C);
    dim3 block(A, N_f4);

    float* h_source = (float*)malloc(B * S * C * A * N * sizeof(float));
    float* h_dst = (float*)malloc(C * B * A * S * N * sizeof(float));
    float* d_source = NULL;
    float* d_dst = NULL;
    cudaMalloc((void**)&d_source, B * S * C * A * N * sizeof(float));
    cudaMalloc((void**)&d_dst, C * B * A * S * N * sizeof(float));
    _init_host_buffer(h_source);
    cpu_transform(h_source, h_dst);
    print(h_dst);
    cudaMemcpy(d_source, h_source, B * S * C * A * N * sizeof(float), cudaMemcpyHostToDevice);
    transform<<<grid, block>>>((float4*)d_source, (float4*)d_dst);
    cudaMemcpy(h_dst, d_dst, C * B * A * S * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("------After Kernel-----\n");
    print(h_dst);
    cudaFree(d_dst);
    cudaFree(d_source);
    free(h_dst);
    free(h_source);

    return 0;
}
