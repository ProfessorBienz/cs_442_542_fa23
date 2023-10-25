#ifndef MPI_MATMAT_HPP
#define MPI_MATMAT_HPP

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// Return process in process-row 'row' and
// process-column 'col'
int get_proc(int row, int col, int sq_procs);
void get_init_procs(int rank_row, int rank_col, int sq_num_procs,
        int* send_proc_A, int* send_proc_B, int* recv_proc_A, int* recv_proc_B);
void get_rotation_procs(int rank_row, int rank_col, int sq_num_procs,
        int* send_proc_A, int* send_proc_B, int* recv_proc_A, int* recv_proc_B);
void swap(float** send_A, float** recv_A, float** send_B, float** recv_B);
void communicate(int send_proc, int recv_proc,
        int tag, int size, int send_first,
        float* sendbuf, float* recvbuf);
void cuda_aware_comm(int send_proc, int recv_proc,
        int tag, int size, int send_first, cudaMemcpyKind direction,
        float* sendbuf, float* recvbuf);
void matmat(int n, float* A, float* B, float* C);
__global__ void matrixMultKernel(int n, float* A, float* B, float* C);


// Cannon's Algorithm; To Be Written By You
//     Rotates chunks of A right through row
//     Rotates chunks of B down column
//     Add this method to 'cannon.cpp'
void mpi_cannon(float* A, float* B, float* C, 
        int n, int sq_num_procs,int rank_row, int rank_col);
void cuda_aware_cannon(float* d_A, float* d_B, float* d_C,
        int n, int sq_num_procs, int rank_row, int rank_col);
void copy_to_cpu_cannon(float* d_A, float* d_B, float* d_C,
        int n, int sq_num_procs, int rank_row, int rank_col);

#endif
