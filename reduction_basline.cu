#include <iostream>
#include <cstdlib>
const int Len = 50000;
const int THREADS_PER_BLOCK = 128;

__global__ void reduction(float* p_source, float* p_destination)
{
    // 计算当前线程块负责的数组区域的起始地址
    float* p_target = p_source + (blockIdx.x * blockDim.x + threadIdx.x);

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            *p_target += *(p_target + offset);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        p_destination[blockIdx.x] = *p_target;
    }
}

void _init_array(float* p)
{
    float sum = 0;
    for (int i = 0; i < Len; i++)
    {
        float tmp = 1.0 * rand() / RAND_MAX;
        p[i] = tmp;
        sum += tmp;
    }
    std::cout << "Target Sum: " << sum << std::endl;
}

void _get_gpu_result(float* p)
{
    float sum = 0;
    for (int i = 0; i < (Len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; i++)
    {
        sum += p[i];
    }
    std::cout << "GPU Sum: " << sum << std::endl;
}

int main(void)
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float* h_src = (float*)malloc(Len * sizeof(float));
    float* h_des = (float*)malloc((Len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * sizeof(float));
    _init_array(h_src);
    float* d_src;
    float* d_des;
    cudaMalloc((void**)&d_src, Len * sizeof(float));
    cudaMalloc((void**)&d_des, (Len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * sizeof(float));
    cudaMemcpy(d_src, h_src, Len * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    reduction<<<(Len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_src, d_des);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaMemcpy(h_des, d_des, (Len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * sizeof(float), cudaMemcpyDeviceToHost);
    _get_gpu_result(h_des);
    cudaFree(d_des);
    cudaFree(d_src);
    free(h_des);
    free(h_src);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    std::cout << "Elapsed Time: " << elapsed_time << " ms" << std::endl;
    cudaEventDestroy(end);
    cudaEventDestroy(start);
    return 0;
}

