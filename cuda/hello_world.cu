#include <stdio.h>
#include <cuda.h>

__global__ void print_kernel()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    print_kernel<<<10,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
