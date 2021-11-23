#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
const float KB = 1024.0;
const float MB = 1024.0 * 1024.0;

int main(void)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float l2_size = static_cast<float>(prop.l2CacheSize) / MB;
    float p_size = static_cast<float>(prop.persistingL2CacheMaxSize) / MB;
    std::cout << "L2 Cache Size: " << l2_size << " MB" << std::endl;
    std::cout << "Persisting L2 Cache Max Size: " << p_size << " MB" << std::endl;
    float shared_size = static_cast<float>(prop.sharedMemPerBlock) / KB;
    std::cout << "Max Shared Memory Per Block: " << shared_size << "KB" << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    return 0;
}

