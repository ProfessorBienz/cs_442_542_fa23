// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.

#include "gtest/gtest.h"
#include "mpi_cannon.hpp"

void matmat(int n, float* A, float* B, float* C)
{   
    float val; 
    for (int i = 0; i < n; i++)
    {   
        for (int j = 0; j < n; j++)
        {   
            val = C[i*n+j]; 
            for (int k = 0; k < n; k++)
                val += A[i*n+k] * B[k*n+j];
            C[i*n+j] = val;
        }
    }
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(CollectiveTest, TestsInCollective)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int N = 128;
    int sq_num_procs = sqrt(num_procs);
    int rank_row = rank / sq_num_procs;
    int rank_col = rank % sq_num_procs;
    int n = N / sq_num_procs;
    int size = n*n;

    float *h_A, *h_B, *h_C_CPU, *h_C_cuda, *h_C_copy;
    h_A = new float[size];
    h_B = new float[size];
    h_C_CPU = new float[size];
    h_C_cuda = new float[size];
    h_C_copy = new float[size];

/*    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size*sizeof(float));
    cudaMalloc((void**)&d_B, size*sizeof(float));
    cudaMalloc((void**)&d_C, size*sizeof(float));
*/
    for (int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            h_A[i*n+j] = 1.0/(rank*n*n + i*n + j+1);
            h_B[i*n+j] = 1.0/(rank*n*n + i*n + j+1);
        }
    }
    
    mpi_cannon(h_A, h_B, h_C_CPU, n, sq_num_procs, rank_row, rank_col);

cuda_aware_cannon(h_A, h_B, h_C_cuda, n, sq_num_procs, rank_row, rank_col);
copy_to_cpu_cannon(h_A, h_B, h_C_copy, n, sq_num_procs, rank_row, rank_col);

/*
    cudaMemcpy(d_A, h_A, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size*sizeof(float), cudaMemcpyHostToDevice);
    //cuda_aware_cannon(d_A, d_B, d_C, n, sq_num_procs, rank_row, rank_col);
    cudaMemcpy(h_C_cuda, d_C, size*sizeof(float), cudaMemcpyDeviceToHost);


    cudaMemcpy(d_A, h_A, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size*sizeof(float), cudaMemcpyHostToDevice);
    //copy_to_cpu_cannon(d_A, d_B, d_C, n, sq_num_procs, rank_row, rank_col);
    cudaMemcpy(h_C_copy, d_C, size*sizeof(float), cudaMemcpyDeviceToHost);
*/
    for (int i = 0; i < size; i++)
    {
    //    ASSERT_NEAR(h_C_CPU[i], h_C_cuda[i], 1e-10);
        ASSERT_NEAR(h_C_CPU[i], h_C_copy[i], 1e-5);
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_CPU;
    delete[] h_C_cuda;
    delete[] h_C_copy;
/*
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
*/
} // end of  TEST(ParStrengthTest, TestsInTests) //
