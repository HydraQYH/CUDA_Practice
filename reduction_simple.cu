#include <stdio.h>
#include "cuda.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ int reduction_core(cg::thread_group g, int *temp, int val)
{
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if(lane<i) val += temp[lane + i];
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
}

__global__ void reduction(void)
{
    __shared__ int buffer[32];
    cg::thread_block b = cg::this_thread_block();
    // cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    // int sum = reduction_core(g, buffer, threadIdx.x);
    int sum = reduction_core(b, buffer, threadIdx.x);

    printf("Thread ID: %d, value: %d\n", threadIdx.x, sum);
}

int main(void)
{
    reduction<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}