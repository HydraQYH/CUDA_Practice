#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
const int WARP_SZIE = 32;
const int GRID = 4096;
const int BLOCK = WARP_SZIE * WARP_SZIE;

__global__ void reduce_kernel_1(float* p)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);
    int gsize = g.size();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = p[tid];
    float max_val = val;
    // reduce: sum
    for (int i = 1; i < gsize; i *= 2) {
        auto temp = g.shfl_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }
    p[tid] = max_val;
}

__global__ void reduce_kernel_2(float* p)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);
    int gsize = g.size();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = p[tid];
    float max_val = val;
    // reduce: sum
    max_val = cg::reduce(g, val, cg::greater<float>());
    p[tid] = max_val;
}

int main(void)
{
    float* d_p = nullptr;
    float* h_p = (float*)malloc(GRID * BLOCK * sizeof(float));
    cudaMalloc((void**)&d_p, GRID * BLOCK * sizeof(float));
    for (size_t i = 0; i < GRID * BLOCK; i++)
    {
        h_p[i] = static_cast<float>(i);
    }
    
    cudaMemcpy((void*)d_p, (void*)h_p, GRID * BLOCK * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    for (size_t i = 0; i < 1000; i++)
    {
        if (i == 50)
        {
            cudaEventRecord(start);
        }
        // reduce_kernel_1<<<GRID, BLOCK>>>(d_p);
        reduce_kernel_2<<<GRID, BLOCK>>>(d_p);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaMemcpy((void*)h_p, (void*)d_p, GRID * BLOCK * sizeof(float), cudaMemcpyDeviceToHost);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    std::cout << "Elapsed Time: " << elapsed_time << " ms" << std::endl;

    cudaEventDestroy(end);
    cudaEventDestroy(start);
    
    cudaFree(d_p);
    free(h_p);
    return 0;
}

