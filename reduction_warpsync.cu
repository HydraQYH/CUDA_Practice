#include <iostream>
#include <cstdlib>
const int Len = 50000;
const int THREADS_PER_BLOCK = 128;

__global__ void reduction(float* p_source, float* p_destination)
{
    // 声明共享内存区域 大小为THREADS_PER_BLOCK个float
    // __shared__ float s_d[THREADS_PER_BLOCK];
    extern __shared__ float s_d[];

    // 从全局内存向共享内存中拷贝数据
    float* p_target = p_source + (blockIdx.x * blockDim.x + threadIdx.x);
    s_d[threadIdx.x] = *p_target;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            s_d[threadIdx.x] += s_d[threadIdx.x + offset];
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            s_d[threadIdx.x] += s_d[threadIdx.x + offset];
        }
        __syncwarp();
    }

    if (threadIdx.x == 0)
    {
        // p_destination[blockIdx.x] = s_d[0];
        atomicAdd(p_destination, s_d[0]);
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

int main(void)
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float* h_src = (float*)malloc(Len * sizeof(float));
    float* h_des = (float*)malloc(sizeof(float));
    *h_des = 0.0;
    _init_array(h_src);
    float* d_src;
    float* d_des;
    cudaMalloc((void**)&d_src, Len * sizeof(float));
    cudaMalloc((void**)&d_des, sizeof(float));
    cudaMemcpy(d_src, h_src, Len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_des, h_des, sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    // reduction<<<(Len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_src, d_des);
    reduction<<<(Len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(d_src, d_des);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaMemcpy(h_des, d_des, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU Sum: " << *h_des << std::endl;
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

