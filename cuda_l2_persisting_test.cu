#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
const int MB = 1024 * 1024;
const int THREADS = 512;
const int FLOAT_DATA_SIZE = sizeof(float);

__global__ void kernel(float *data_persistent, float *data_streaming, int dataSize, int freqSize) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    
    /*Each CUDA thread accesses one element in the persistent data section
      and one element in the streaming data section.
      Because the size of the persistent memory region (freqSize * sizeof(int) bytes) is much 
      smaller than the size of the streaming memory region (dataSize * sizeof(int) bytes), data 
      in the persistent region is accessed more frequently*/

    data_persistent[tid % freqSize] = 2 * data_persistent[tid % freqSize]; 
    data_streaming[tid % dataSize] = 2 * data_streaming[tid % dataSize];
}     

int main(void)
{
    // 创建一块内存区域 大小64MB
    float* d_data = nullptr;
    cudaMalloc((void**)&d_data, 64 * MB);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    // 这64MB的数据 每个Thread访问一个float stream数据
    dim3 grid(64 * MB / FLOAT_DATA_SIZE / THREADS);
    dim3 block(THREADS);
    // 设置前4M的L2缓存 为persistent
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t size = min(4 * MB, prop.persistingL2CacheMaxSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
    cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(d_data); // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = 4 * MB;                    // Number of bytes for persisting accesses. (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)                                                                
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                          // Hint for L2 cache hit ratio for persisting accesses in the num_bytes region
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);  //Set the attributes to a CUDA stream of type cudaStream_t

    cudaEventRecord(start);
    for (size_t i = 0; i < 100; i++)
    {
      kernel<<<grid, block>>>(d_data, d_data, 64 * MB / FLOAT_DATA_SIZE, 4 * MB / FLOAT_DATA_SIZE);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    std::cout << "Elapsed Time: " << elapsed_time << " ms" << std::endl;
    
    cudaEventDestroy(end);
    cudaEventDestroy(start);
    cudaFree(d_data);
    return 0;
}

