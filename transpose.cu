#include <iostream>
#include <cstdlib>
const int SOURCE_ROW = 10384;
const int SOURCE_COLUMN = 10259;
const int DESTINATION_ROW = SOURCE_COLUMN;
const int DESTINATION_COLUMN = SOURCE_ROW;

const int CHUNK = 32;

__global__ void transpose(float* p_src, float* p_des)
{
    int continuous_index = blockIdx.x * blockDim.x + threadIdx.x;   // 目标矩阵列索引
    int no_continuous_index = blockIdx.y * blockDim.y + threadIdx.y;    // 目标矩阵行索引

    if (continuous_index < DESTINATION_COLUMN && no_continuous_index < DESTINATION_ROW)
    {
        // no_continuous_index < SOURCE_COLUMN && continuous_index < SOURCE_ROW
        p_des[no_continuous_index * DESTINATION_COLUMN + continuous_index] = p_src[continuous_index * SOURCE_COLUMN + no_continuous_index];
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
    // 计算线程格
    dim3 grid((DESTINATION_COLUMN + CHUNK - 1) / CHUNK, (DESTINATION_ROW + CHUNK - 1) / CHUNK);
    dim3 block(CHUNK, CHUNK);
    cudaEventRecord(start);
    transpose<<<grid, block>>>(d_src, d_des);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaMemcpy(h_des, d_des, DESTINATION_ROW * DESTINATION_COLUMN * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "-----Transpose Result-----" << std::endl;
    // printMatrix(h_des, DESTINATION_ROW, DESTINATION_COLUMN);
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

