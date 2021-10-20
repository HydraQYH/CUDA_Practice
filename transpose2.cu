#include <iostream>
#include <cstdlib>
const int SOURCE_ROW = 10384;
const int SOURCE_COLUMN = 10259;
const int DESTINATION_ROW = SOURCE_COLUMN;
const int DESTINATION_COLUMN = SOURCE_ROW;

const int CHUNK = 32;

__global__ void transpose(float* p_src, float* p_des)
{
    __shared__ float chunk[CHUNK][CHUNK + 1];
    // 计算当前线程要处理的源矩阵元素的坐标
    int src_col = blockIdx.x * CHUNK + threadIdx.x;
    int src_row = blockIdx.y * CHUNK + threadIdx.y;
    if (src_col < SOURCE_COLUMN && src_row < SOURCE_ROW)
    {
        // 从全局内存向共享内存拷贝数据 全局内存的读取是连续的（threadIdx.x相邻的线程读取连续的内存）
        chunk[threadIdx.y][threadIdx.x] = p_src[src_row * SOURCE_COLUMN + src_col];
    }
    __syncthreads();

    // 写入目标矩阵（写入不连续的情况）
    // 原方案中 将源矩阵(i, j)位置元素存入共享内存(i % 32, j % 32) 并将该元素存入目标矩阵(j, i)位置
    // int des_col = src_row;  // blockIdx.y * CHUNK + threadIdx.y
    // int des_row = src_col;  // blockIdx.x * CHUNK + threadIdx.x
    // if (des_col < DESTINATION_COLUMN && des_row < DESTINATION_ROW)
    // {
    //     p_des[des_row * DESTINATION_COLUMN + des_col] = chunk[threadIdx.y][threadIdx.x];
    // }
    
    // 优化方案中 将源矩阵(i, j)位置元素存入共享内存(i % 32, j % 32)
    // 令i_ = i - i % 32 j_ = j - j % 32
    // 从共享内存中(j % 32, i % 32)位置提取元素 写入目标矩阵(j_ + i % 32, i_ + j % 32)
    int des_col = blockIdx.y * CHUNK + threadIdx.x;
    int des_row = blockIdx.x * CHUNK + threadIdx.y;
    if (des_col < DESTINATION_COLUMN && des_row < DESTINATION_ROW)
    {
        p_des[des_row * DESTINATION_COLUMN + des_col] = chunk[threadIdx.x][threadIdx.y];
    }
}

void _matrix_initialize(float* p)
{
    for (int i = 0; i < SOURCE_ROW; i++)
    {
        for (int j = 0; j < SOURCE_COLUMN; j++)
        {
            p[i * SOURCE_COLUMN + j] = rand() * 1.0 / RAND_MAX;
        }
    }
}

void printMatrix(float* p, int row, int column)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            std::cout << p[i * column + j] << ' ';
        }
        std::cout << std::endl;
    }
}

int main(void)
{
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    // 分配矩阵主机存储空间
    float* h_src = (float*)malloc(SOURCE_ROW * SOURCE_COLUMN * sizeof(float));
    float* h_des = (float*)malloc(DESTINATION_ROW * DESTINATION_COLUMN * sizeof(float));
    // 分配设备内存
    float* d_src = nullptr;
    float* d_des = nullptr;
    cudaMalloc((void**)&d_src, SOURCE_ROW * SOURCE_COLUMN * sizeof(float));
    cudaMalloc((void**)&d_des, DESTINATION_ROW * DESTINATION_COLUMN * sizeof(float));
    _matrix_initialize(h_src);
    // printMatrix(h_src, SOURCE_ROW, SOURCE_COLUMN);
    cudaMemcpy(d_src, h_src, SOURCE_ROW * SOURCE_COLUMN * sizeof(float), cudaMemcpyHostToDevice);
    // 计算线程格 线程格大小与源矩阵对应
    dim3 grid((SOURCE_COLUMN + CHUNK - 1) / CHUNK, (SOURCE_ROW + CHUNK - 1) / CHUNK);
    dim3 block(CHUNK, CHUNK);
    cudaEventRecord(start);
    transpose<<<grid, block>>>(d_src, d_des);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaMemcpy(h_des, d_des, DESTINATION_ROW * DESTINATION_COLUMN * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "-----Transpose Result-----" << std::endl;
    // printMatrix(h_des, DESTINATION_ROW, DESTINATION_COLUMN);
    cudaDeviceSynchronize();
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

