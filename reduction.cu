#include <stdio.h>
#include "cuda.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void reduction(void)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int max_val = threadIdx.x;
    for (int i = 1; i < 32; i *= 2) {
        int temp = g.shfl_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    printf("Thread ID: %d, value: %d\n", threadIdx.x, max_val);
}

int main(void)
{
    reduction<<<1, 128>>>();
    cudaDeviceSynchronize();
    return 0;
}