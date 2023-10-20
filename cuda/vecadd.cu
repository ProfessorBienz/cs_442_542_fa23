#include <cuda.h>
#include "../timer.h"

__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void vecAdd(float* A, float* B, float* C, int n)
{
    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

double sum(int n, float* h_C)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        s += h_C[i];
    return s;
}

int main(int argc, char* argv[])
{
    float *h_A, *h_B, *h_C;
    double t0, tfinal;

    if (argc <= 1)
    {
        printf("Pass Vector Size as Command Line Arg\n");
        return 0;
    }

    int n = atoi(argv[1]);

    cudaMallocHost((void**)&h_A, n*sizeof(float));
    cudaMallocHost((void**)&h_B, n*sizeof(float));
    cudaMallocHost((void**)&h_C, n*sizeof(float));

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n*sizeof(float));
    cudaMalloc((void**)&d_B, n*sizeof(float));
    cudaMalloc((void**)&d_C, n*sizeof(float));

    for (int i = 0; i < n; i++)
    {
        h_A[i] = 0.5;
        h_B[i] = 0.7;
    }

    t0 = get_time();
    vecAdd(h_A, h_B, h_C, n);
    tfinal = get_time() - t0;
    printf("VecAdd Time %e, Sum %e\n", tfinal, sum(n, h_C));


    // Copy host array to device array
    t0 = get_time();
    cudaMemcpy(d_A, h_A, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n*sizeof(float), cudaMemcpyHostToDevice);

    // Launch GPU Kernel
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, n*sizeof(float), cudaMemcpyDeviceToHost);
    tfinal = get_time() - t0;

    printf("VecAddKernel Time %e, Sum %e\n", tfinal, sum(n, h_C));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
