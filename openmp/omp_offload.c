#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include <omp.h>

// module load gcc/10.2.0-7uu2
// gcc -o matmat_omp matmat_omp.c -O2 -fopenmp

void test_serial(int n, double* A, double* B, double* C, int n_iter)
{
    double val;
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                val = 0;
                for (int k = 0; k < n; k++)
                {
                    val += A[i*n+k] * B[k*n+j];
                }
                C[i*n+j] = val;
            }
        }
    }
}

void test_omp(int n, double* A, double* B, double* C, int n_iter)
{
    double val;
    for (int iter = 0; iter < n_iter; iter++)
    {
        #pragma omp parallel for private(val)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                val = 0;
                for (int k = 0; k < n; k++)
                {
                    val += A[i*n+k] * B[k*n+j];
                }
                C[i*n+j] = val;
            }
        }
    }
}

void test_omp_gpu(int n, double* A, double* B, double* C, int n_iter)
{
    double val;
    int size = n*n;
    for (int iter = 0; iter < n_iter; iter++)
    {
        #pragma omp target map(tofrom:A[0:size], B[0:size], C[0:size]) 
        {
            #pragma omp parallel for private(val)
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    val = 0;
                    for (int k = 0; k < n; k++)
                    {
                        val += A[i*n+k] * B[k*n+j];
                    }
                    C[i*n+j] = val;
                }
            }
        }
    }
}

void test_omp_gpu_tofrom(int n, double* A, double* B, double* C, int n_iter)
{
    double val;
    int size = n*n;
    for (int iter = 0; iter < n_iter; iter++)
    {
        #pragma omp target map(to:A[0:size], B[0:size]) map(from:C[0:size])
        {
            #pragma omp parallel for private(val)
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    val = 0;
                    for (int k = 0; k < n; k++)
                    {
                        val += A[i*n+k] * B[k*n+j];
                    }
                    C[i*n+j] = val;
                }
            }
        }
    }
}

void test_omp_gpu_data(int n, double* A, double* B, double* C, int n_iter)
{
    double val;
    int size = n*n;
    for (int iter = 0; iter < n_iter; iter++)
    {
        #pragma omp target data map(to:A[0:size], B[0:size]) map(from:C[0:size])
        {
            #pragma omp target 
            #pragma omp parallel for private(val)
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    val = 0;
                    for (int k = 0; k < n; k++)
                    {
                        val += A[i*n+k] * B[k*n+j];
                    }
                    C[i*n+j] = val;
                }
            }
        }
    }
}

void test_omp_gpu_teams(int n, double* A, double* B, double* C, int n_iter)
{
    double val;
    int size = n*n;
    for (int iter = 0; iter < n_iter; iter++)
    {
        #pragma omp target data map(to:A[0:size], B[0:size]) map(from:C[0:size])
        {
            #pragma omp target teams num_teams(n)
            {
                #pragma omp distribute parallel for private(val)
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    { 
                        val = 0;
                        for (int k = 0; k < n; k++)
                        {
                            val += A[i*n+k] * B[k*n+j];
                        }
                        C[i*n+j] = val;
                    }
                }
            }
        }
    }
}

void test_omp_gpu_split(int n, double* A, double* B, double* C, int n_iter)
{
    double val;
    int size = n*n;
    for (int iter = 0; iter < n_iter; iter++)
    {
        #pragma omp target data map(to:A[0:size], B[0:size]) map(from:C[0:size]) 
        {
            #pragma omp target teams num_teams(n) private(val)
            {
                #pragma omp distribute private(val)
                for (int i = 0; i < n; i++)
                {
                    #pragma omp parallel for private(val)
                    for (int j = 0; j < n; j++)
                    {
                        val = 0;
                        for (int k = 0; k < n; k++)
                        {
                            val += A[i*n+k] * B[k*n+j];
                        }
                        C[i*n+j] = val;
                    }
                }
            }
        }
    }
}

void test_omp_gpu_split_rr(int n, double* A, double* B, double* C, int n_iter)
{
    double val;
    int size = n*n;
    for (int iter = 0; iter < n_iter; iter++)
    {
        #pragma omp target data map(to:A[0:size], B[0:size]) map(from:C[0:size])
        {
            #pragma omp target teams num_teams(n) private(val)
            {
                #pragma omp distribute private(val)
                for (int i = 0; i < n; i++)
                {
                    #pragma omp parallel for private(val) schedule(static,1)
                    for (int j = 0; j < n; j++)
                    {
                        val = 0;
                        for (int k = 0; k < n; k++)
                        {
                            val += A[i*n+k] * B[k*n+j];
                        }
                        C[i*n+j] = val;
                    }
                }
            }
        }
    }
}

void test_omp_gpu_collapse(int n, double* A, double* B, double* C, int n_iter)
{
    double val;
    int size = n*n;
    for (int iter = 0; iter < n_iter; iter++)
    {
        #pragma omp target data map(to:A[0:size]) map(to:B[0:size]) map(from:C[0:size])
        {
            #pragma omp target
            #pragma omp teams distribute parallel for collapse(2) num_teams(n) thread_limit(n) private(val)
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    val = 0;
                    for (int k = 0; k < n; k++)
                    {
                        val += A[i*n+k] * B[k*n+j];
                    }
                    C[i*n+j] = val;
                }
            }
        }
    }
}

void test_omp_gpu_collapse_rr(int n, double* A, double* B, double* C, int n_iter)
{
    double val;
    int size = n*n;
    for (int iter = 0; iter < n_iter; iter++)
    {
        #pragma omp target data map(to:A[0:size]) map(to:B[0:size]) map(from:C[0:size])
        {
            #pragma omp target
            #pragma omp teams distribute parallel for collapse(2) num_teams(n) thread_limit(n) private(val) schedule(static,1)
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    val = 0;
                    for (int k = 0; k < n; k++)
                    {
                        val += A[i*n+k] * B[k*n+j];
                    }
                    C[i*n+j] = val;
                }
            }
        }
    }
}


double sum(int n, double* C)
{
    int i, j;
    double s = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            s += C[i*n+j];
        }
    }
    return s;
}

// This program runs matrix matrix multiplication with single pointers
// Test vectorization improvements for both doubles and floats
int main(int argc, char* argv[])
{
    int i, j;
    double start, end;
    int n = atoi(argv[1]);
    int n_iter = 1;
    int size = n*n;

    double* A = (double*)malloc(size*sizeof(double));
    double* B = (double*)malloc(size*sizeof(double));
    double* C = (double*)malloc(size*sizeof(double));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i*n+j] = 1.0 / (i*n+j+1);
            B[i*n+j] = 1.0;
            C[i*n+j] = 0.0;
        }
    }


/*
    start = get_time();
    test_serial(n, A, B, C, n_iter);
    end = get_time();
    printf("Serial: Sum %e, Time Per MatMat %e\n", sum(n, C), (end - start)/n_iter);

    start = get_time();
    test_omp(n, A, B, C, n_iter);
    end = get_time();
    printf("CPU OMP: Sum %e, Time Per MatMat %e\n", sum(n, C), (end - start)/n_iter);

    start = get_time();
    test_omp_gpu(n, A, B, C, n_iter);
    end = get_time();
    printf("GPU OMP: Sum %e, Time Per MatMat %e\n", sum(n, C), (end - start) / n_iter);

    start = get_time();
    test_omp_gpu_tofrom(n, A, B, C, n_iter);
    end = get_time();
    printf("GPU OMP To/From: Sum %e, Time Per MatMat %e\n", sum(n, C), (end - start) / n_iter);

    start = get_time();
    test_omp_gpu_data(n, A, B, C, n_iter);
    end = get_time();
    printf("GPU OMP Data: Sum %e, Time Per MatMat %e\n", sum(n, C), (end - start) / n_iter);
*/
    start = get_time();
    test_omp_gpu_teams(n, A, B, C, n_iter);
    end = get_time();
    printf("GPU OMP Teams: Sum %e, Time Per MatMat %e\n", sum(n, C), (end - start) / n_iter);

    start = get_time();
    test_omp_gpu_split(n, A, B, C, n_iter);
    end = get_time();
    printf("GPU OMP Split: Sum %e, Time Per MatMat %e\n", sum(n, C), (end - start) / n_iter);

    start = get_time();
    test_omp_gpu_collapse(n, A, B, C, n_iter);
    end = get_time();
    printf("GPU OMP Collapse: Sum %e, Time Per MatMat %e\n", sum(n, C), (end - start) / n_iter);
    
    start = get_time();
    test_omp_gpu_split_rr(n, A, B, C, n_iter);
    end = get_time();
    printf("GPU OMP Split RR: Sum %e, Time Per MatMat %e\n", sum(n, C), (end - start) / n_iter);

    start = get_time();
    test_omp_gpu_collapse_rr(n, A, B, C, n_iter);
    end = get_time();
    printf("GPU OMP Collapse RR: Sum %e, Time Per MatMat %e\n", sum(n, C), (end - start) / n_iter);
    

    free(A);
    free(B);
    free(C);
        
    return 0;
}

