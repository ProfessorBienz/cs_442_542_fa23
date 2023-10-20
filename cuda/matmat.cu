#include <cuda.h>
#include "../timer.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define TILE_WIDTH 32

__global__ void matrixMultKernel(float* A, float* B, float* C, int n)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (row < n && col < n)
    {
        float val = 0;
        for (int k = 0; k < n; k++)
            val += A[row*n+k] * B[k*n+col];
        C[row*n+col] = val;
    }
}

__global__ void matrixMultTiledKernel(float* A, float* B, float* C, int n)
{
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int col = block_x * blockDim.x + thread_x;
    int row = block_y * blockDim.y + thread_y;

    float val = 0;
    for (int i = 0; i < n / TILE_WIDTH; i++)
    {
        A_shared[thread_y][thread_x] = A[row*n + i*TILE_WIDTH + thread_x];
        B_shared[thread_y][thread_x] = B[(i*TILE_WIDTH + thread_y)*n + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
            val += A_shared[thread_y][k] * B_shared[k][thread_x];
        __syncthreads();
    }

    C[row*n + col] = val;
}

__global__ void matrixMultTiledUnrolledKernel(float* A, float* B, float* C, int n)
{   
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];
    
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x0 = threadIdx.x*2;
    int thread_x1 = threadIdx.x*2 + 1;
    int thread_y0 = threadIdx.y;
    int thread_y1 = threadIdx.y + blockDim.y;
    
    int col0 = block_x * blockDim.x + thread_x0;
    int col1 = block_x * blockDim.x + thread_x1;
    int row0 = block_y * blockDim.y + thread_y0;
    int row1 = block_y * blockDim.y + thread_y1;
    
    float val00 = 0;
    float val01 = 0;
    float val10 = 0;
    float val11 = 0;
    for (int i = 0; i < n / TILE_WIDTH; i++)
    {
        A_shared[thread_y0][thread_x0] = A[row0*n + i*TILE_WIDTH + thread_x0];
        A_shared[thread_y0][thread_x1] = A[row0*n + i*TILE_WIDTH + thread_x1];
        A_shared[thread_y1][thread_x0] = A[row1*n + i*TILE_WIDTH + thread_x0];
        A_shared[thread_y1][thread_x1] = A[row1*n + i*TILE_WIDTH + thread_x1];

        B_shared[thread_y0][thread_x0] = B[(i*TILE_WIDTH + thread_y0)*n + col0];
        B_shared[thread_y0][thread_x1] = B[(i*TILE_WIDTH + thread_y0)*n + col1];
        B_shared[thread_y1][thread_x0] = B[(i*TILE_WIDTH + thread_y1)*n + col0];
        B_shared[thread_y1][thread_x1] = B[(i*TILE_WIDTH + thread_y1)*n + col1];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
        {
            val00 += A_shared[thread_y0][k] * B_shared[k][thread_x0];
            val01 += A_shared[thread_y0][k] * B_shared[k][thread_x1];
            val10 += A_shared[thread_y1][k] * B_shared[k][thread_x0];
            val11 += A_shared[thread_y1][k] * B_shared[k][thread_x1];
        }
        __syncthreads();
    }

    C[row0*n + col0] = val00;
    C[row0*n + col1] = val01;
    C[row1*n + col0] = val10;
    C[row1*n + col1] = val11;
}

void matrixMult(float* A, float* B, float* C, int n)
{
    float val;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            val = 0;
            for (int k = 0; k < n; k++)
                val += A[i*n+k] * B[k*n+j];
            C[i*n+j] = val;
        }
    }
}

double sum(float* C, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            s += C[i*n+j];
    return s;
}


int main(int argc, char* argv[])
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    double t0, tfinal;

    if (argc == 1)
    {
        printf("Pass Matrix Dimension as Command Line Arg\n");
        return 0;
    }
    
    int n = atoi(argv[1]);
    int size = n*n*sizeof(float);
    
    cudaMallocHost((void**)&h_A, size);
    cudaMallocHost((void**)&h_B, size);
    cudaMallocHost((void**)&h_C, size);
    for (int i = 0; i < n*n; i++)
    {
        h_A[i] = 0.5;
        h_B[i] = 0.2;
    }   
    
    t0 = get_time();
    matrixMult(h_A, h_B, h_C, n);
    tfinal = get_time() - t0;
    printf("MatrixMult Time %e, Sum %e\n", tfinal, sum(h_C, n));
    
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    

    // Matmat
    dim3 dimBlock(32,32);
    int grid_dim = ceil(n / 32.0);
    dim3 dimGrid(grid_dim, grid_dim);
    
    t0 = get_time();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    matrixMultKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    tfinal = get_time() - t0;
    printf("MatrixMultKernel Time %e, Size %e\n", tfinal, sum(h_C, n));
    
    t0 = get_time();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    matrixMultTiledKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    tfinal = get_time() - t0;
    printf("MatrixMultTiledKernel Time %e, Size %e\n", tfinal, sum(h_C, n));


    dim3 dimBlockUnroll(16, 16);
    int grid_dim_unroll = ceil(n / 32.0);
    dim3 dimGridUnroll(grid_dim_unroll, grid_dim_unroll);
    t0 = get_time();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    matrixMultTiledUnrolledKernel<<<dimGridUnroll, dimBlockUnroll>>>(d_A, d_B, d_C, n);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    tfinal = get_time() - t0;
    printf("MatrixMultTiledKernel Time %e, Size %e\n", tfinal, sum(h_C, n));


    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0;
    int l_dim = n;

    t0 = get_time();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A,
            l_dim, d_B, l_dim, &beta, d_C, l_dim);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    tfinal = get_time() - t0;
    printf("CUBLAS SGEMM Time %e, Size %e\n", tfinal, sum(h_C, n));

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    
}




