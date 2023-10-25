#include "mpi_cannon.hpp"

__global__ void matrixMultKernel(int n, float* A, float* B, float* C)
{   
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    float val;
    
    if (row < n && col < n)
    {   
        val = C[row*n+col];
        for (int k = 0; k < n; k++)
            val += A[row*n+k] * B[k*n+col];
        C[row*n+col] = val;
    }
}


void matmat(int n, float* A, float* B, float* C)
{
    float val;
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            val = C[row*n+col];
            for (int k = 0; k < n; k++)
                val += A[row*n+k] * B[k*n+col];
            C[row*n+col] = val;
        }
    }
}


void communicate(int send_proc, int recv_proc,
        int tag, int size, int send_first,
        float* sendbuf, float* recvbuf)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (send_proc == rank)
    {
        memcpy(recvbuf, sendbuf, size*sizeof(float));
    }
    else if (send_first)
    {
        MPI_Send(sendbuf, size, MPI_FLOAT, send_proc, tag, MPI_COMM_WORLD);
        MPI_Recv(recvbuf, size, MPI_FLOAT, recv_proc, tag, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
    }
    else
    {
        MPI_Recv(recvbuf, size, MPI_FLOAT, recv_proc, tag, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        MPI_Send(sendbuf, size, MPI_FLOAT, send_proc, tag, MPI_COMM_WORLD);
    }
}

void cuda_aware_comm(int send_proc, int recv_proc,
        int tag, int size, int send_first, cudaMemcpyKind direction,
        float* sendbuf, float* recvbuf)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (send_proc == rank)
    {
        cudaMemcpy(recvbuf, sendbuf, size*sizeof(float), direction);
    }
    else if (send_first)
    {
        MPI_Send(sendbuf, size, MPI_FLOAT, send_proc, tag, MPI_COMM_WORLD);
        MPI_Recv(recvbuf, size, MPI_FLOAT, recv_proc, tag, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
    }
    else
    {
        MPI_Recv(recvbuf, size, MPI_FLOAT, recv_proc, tag, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        MPI_Send(sendbuf, size, MPI_FLOAT, send_proc, tag, MPI_COMM_WORLD);
    }
}

// Return process in process-row 'row' and
// process-column 'col'
int get_proc(int row, int col, int sq_procs)
{
    return row*sq_procs + col;
}

void get_init_procs(int rank_row, int rank_col, int sq_num_procs,
        int* send_proc_A, int* send_proc_B, int* recv_proc_A, int* recv_proc_B)
{   
    *send_proc_A = get_proc(rank_row, rank_col-rank_row, sq_num_procs);
    *send_proc_B = get_proc(rank_row-rank_col, rank_col, sq_num_procs);
    *recv_proc_A = get_proc(rank_row, rank_col+rank_row, sq_num_procs);
    *recv_proc_B = get_proc(rank_row+rank_col, rank_col, sq_num_procs);
    if (rank_col+rank_row >= sq_num_procs)
    {   
        *recv_proc_A = get_proc(rank_row, rank_col+rank_row-sq_num_procs, sq_num_procs);
        *recv_proc_B = get_proc(rank_row+rank_col-sq_num_procs, rank_col, sq_num_procs);
    }
    if (rank_col - rank_row < 0)
        *send_proc_A = get_proc(rank_row, rank_col-rank_row+sq_num_procs, sq_num_procs);
    if (rank_row - rank_col < 0)
        *send_proc_B = get_proc(rank_row-rank_col+sq_num_procs, rank_col, sq_num_procs);

}

void get_rotation_procs(int rank_row, int rank_col, int sq_num_procs,
        int* send_proc_A, int* send_proc_B, int* recv_proc_A, int* recv_proc_B)
{
    *send_proc_A = get_proc(rank_row, rank_col+1, sq_num_procs);
    *send_proc_B = get_proc(rank_row+1, rank_col, sq_num_procs);
    *recv_proc_A = get_proc(rank_row, rank_col-1, sq_num_procs);
    *recv_proc_B = get_proc(rank_row-1, rank_col, sq_num_procs);

    if (rank_col == sq_num_procs-1)
        *send_proc_A = get_proc(rank_row, 0, sq_num_procs);
    if (rank_row == sq_num_procs-1)
        *send_proc_B = get_proc(0, rank_col, sq_num_procs);
    if (rank_col == 0)
        *recv_proc_A = get_proc(rank_row, sq_num_procs-1, sq_num_procs);
    if (rank_row == 0)
        *recv_proc_B = get_proc(sq_num_procs-1, rank_col, sq_num_procs);
}

void swap(float** send_A, float** recv_A, float** send_B, float** recv_B)
{
    float* tmp;
    tmp = *send_A;
    *send_A = *recv_A;
    *recv_A = tmp;

    tmp = *send_B;
    *send_B = *recv_B;
    *recv_B = tmp;
}




