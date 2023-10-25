#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#include "mpi_cannon.hpp"

// Main Method : 
//     Splits processes into a process grid
//         - rank_row : row of process in process grid
//         - rank_col : column of process in process grid
//     Creates three local matrices, A, B, and C
//     Times all three implementations of parallel DGEMM
//     Prints timings of methods
int main(int argc, char* argv[])
{

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get rank of process and number of processes
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Make sure matrix dimension is in argv
    if (argc <= 1)
    {   
        if (rank == 0)
            printf("Pass Matrix Dimension as Command Line Argument!\n");
        MPI_Finalize();
        return 1;
    }

    // Grab global dimension of matrices (A, B, C)
    int N = atoi(argv[1]);

    // Calculate how many process rows/cols in process-grid
    int sq_num_procs = sqrt(num_procs);
    if (sq_num_procs*sq_num_procs != num_procs)
    {
        if (rank == 0) 
            printf("Number of processes needs to be a square\n");
        MPI_Finalize();
        return 1;
    }

    // Calculate variables
    // - rank_row : process row
    // - rank_col : process col
    // - n : local (per-process) matrix dimension
    int rank_row = rank / sq_num_procs;
    int rank_col = rank % sq_num_procs;
    int n = N / sq_num_procs;
    int size = n*n;

    if (n*n*num_procs != N*N)
    {
        if (rank == 0) 
            printf("Cannot evenly split %d rows and cols over %d processes\n",
                    N, num_procs);
        MPI_Finalize();
        return 1;
    }

    // Allocate three local matrices (A, B, C)
    float *h_A, *h_B, *h_C;
    h_A = new float[size];
    h_B = new float[size];
    h_C = new float[size];

    // Initialize matrices A and B 
    int first_i = rank_row*N;
    int first_j = rank_col;
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            h_A[i*n+j] = 1.0 / (((rank_row*n)+i)*N + (rank_col*n)+j+1);
            h_B[i*n+j] = 1.0 / (((rank_row*n)+i)*N + (rank_col*n)+j+1);
        }
    }
    
    float start, end;

    // Time Cannon's Method
    mpi_cannon(h_A, h_B, h_C, n, sq_num_procs, rank_row, rank_col);
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    mpi_cannon(h_A, h_B, h_C, n, sq_num_procs, rank_row, rank_col);
    end = MPI_Wtime() - start;
    MPI_Reduce(&end, &start, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Cannon's Method on CPU: Elapsed Time %e\n", start);

    // Time CUDA-Aware Cannon's Method
    cuda_aware_cannon(h_A, h_B, h_C, n, sq_num_procs, rank_row, rank_col);
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    cuda_aware_cannon(h_A, h_B, h_C, n, sq_num_procs, rank_row, rank_col);
    end = MPI_Wtime() - start;
    MPI_Reduce(&end, &start, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("CUDA-Aware Cannon's Method: Elapsed Time %e\n", start);

    // Time Copy-to-CPU Cannon's Method
    copy_to_cpu_cannon(h_A, h_B, h_C, n, sq_num_procs, rank_row, rank_col);
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    copy_to_cpu_cannon(h_A, h_B, h_C, n, sq_num_procs, rank_row, rank_col);
    end = MPI_Wtime() - start;
    MPI_Reduce(&end, &start, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Copy-to-CPU Cannon's Method: Elapsed Time %e\n", start);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    MPI_Finalize();
    return 0;
}
