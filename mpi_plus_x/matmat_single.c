#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

// Serial matrix-matrix multiplication
void matmat(int n, double* A, double* B, double* C)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int i, j, k;

    double val;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            val = A[i*n+j];
            for (k = 0; k < n; k++)
            {
                C[i*n+k] += val * B[j*n+k];
            }
        }
    }
 
}

// Shift A 'rank_row' columns
// Shift B 'rank_col' rows
// All pairs of A and B on a single process should be multiplied
// Then, send submatrix of A to neighboring process (rowwise)
// and submatrix of B to neighboring process (columnwise)
void cannon(int n, double* A, double* B, double** C_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int i, j;

    double* C = (double*)malloc(n*n*sizeof(double));
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            C[i*n+j] = 0;
        }
    }

    // Define other matrices to hold A and B
    double* A2 = (double*)malloc(n*n*sizeof(double));
    double* B2 = (double*)malloc(n*n*sizeof(double));
    double* A3 = (double*)malloc(n*n*sizeof(double));
    double* B3 = (double*)malloc(n*n*sizeof(double));

    double* send_A = A;
    double* send_B = B;
    double* recv_A = A2;
    double* recv_B = B2;
    double* tmp;

    int sq_num_procs = sqrt(num_procs);
    int rank_row = rank / sq_num_procs;
    int rank_col = rank % sq_num_procs;

    int proc, shift;
    int proc_row, proc_col;
    int tag_a = 1234;
    int tag_b = 4321;

    MPI_Request send_req_a, send_req_b, recv_req_a, recv_req_b;

    // Cannon Shift:
    // Recv A
    shift = rank_row;
    proc_col = rank_col - shift;
    if (proc_col < 0) proc_col += sq_num_procs;
    proc = rank_row * sq_num_procs + proc_col;
    MPI_Irecv(recv_A, n*n, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, &recv_req_a);
    
    // Recv B
    shift = rank_col;
    proc_row = rank_row - shift;
    if (proc_row < 0) proc_row += sq_num_procs;
    proc = proc_row * sq_num_procs + rank_col;
    MPI_Irecv(recv_B, n*n, MPI_DOUBLE, proc, tag_b, MPI_COMM_WORLD, &recv_req_b);

    // Send A 
    shift = rank_row;
    proc_col = rank_col + shift;
    if (proc_col >= sq_num_procs) proc_col -= sq_num_procs;
    proc = rank_row * sq_num_procs + proc_col;
    MPI_Isend(send_A, n*n, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, &send_req_a);

    // Send B
    shift = rank_col;
    proc_row = rank_row + shift;
    if (proc_row >= sq_num_procs) proc_row -= sq_num_procs;
    proc = proc_row * sq_num_procs + rank_col;
    MPI_Isend(send_B, n*n, MPI_DOUBLE, proc, tag_b, MPI_COMM_WORLD, &send_req_b);

    MPI_Wait(&send_req_a, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_req_a, MPI_STATUS_IGNORE);
    MPI_Wait(&send_req_b, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_req_b, MPI_STATUS_IGNORE);

    tag_a++;
    tag_b++;

    // After initial shift, can multiply pairs of matrices
    matmat(n, recv_A, recv_B, C);

    recv_A = A3;
    recv_B = B3;
    send_A = A2;
    send_B = B2;

    int n_shifts = sq_num_procs - 1;
    for (i = 0; i < n_shifts; i++)
    {
        // Recv A from neighbor
        proc_col = rank_col - 1;
        if (proc_col < 0) proc_col += sq_num_procs;
        proc = rank_row * sq_num_procs + proc_col;
        MPI_Irecv(recv_A, n*n, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, &recv_req_a);
        
        // Recv B from neighbor
        proc_row = rank_row - 1;
        if (proc_row < 0) proc_row += sq_num_procs;
        proc = proc_row * sq_num_procs + rank_col;
        MPI_Irecv(recv_B, n*n, MPI_DOUBLE, proc, tag_b, MPI_COMM_WORLD, &recv_req_b);

        // Send A to neighbor
        proc_col = rank_col + 1;
        if (proc_col >= sq_num_procs) proc_col -= sq_num_procs;
        proc = rank_row * sq_num_procs + proc_col;
        MPI_Isend(send_A, n*n, MPI_DOUBLE, proc, tag_a, MPI_COMM_WORLD, &send_req_a);

        // Send B to neighbor
        proc_row = rank_row + 1;
        if (proc_row >= sq_num_procs) proc_row -= sq_num_procs;
        proc = proc_row * sq_num_procs + rank_col;
        MPI_Isend(send_B, n*n, MPI_DOUBLE, proc, tag_b, MPI_COMM_WORLD, &send_req_b);  

        MPI_Wait(&send_req_a, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req_a, MPI_STATUS_IGNORE);
        MPI_Wait(&send_req_b, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req_b, MPI_STATUS_IGNORE);

        // After each step of communication, multiply locally recvd submatrices
        matmat(n, recv_A, recv_B, C);

        tag_a++;
        tag_b++;

        tmp = send_A;
        send_A = recv_A;
        recv_A = tmp;
        tmp = send_B;
        send_B = recv_B;
        recv_B = tmp;
    }


    free(recv_A);
    free(recv_B);
    free(send_A);
    free(send_B);

    *C_ptr = C;
}

double mat_sum(int n, double* C)
{
    double sum = 0;
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            sum += C[i*n+j];
        }
    }
    return sum;
}

int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int N = atoi(argv[1]);
    int sq_num_procs = sqrt(num_procs);
    int rank_row = rank / sq_num_procs;
    int first_rank_row = rank_row*sq_num_procs;
    int rank_col = rank % sq_num_procs;

    int n = N / sq_num_procs;
    double* A = (double*)malloc(n*n*sizeof(double));
    double* B = (double*)malloc(n*n*sizeof(double));
    double* C;

    srand(rank*time(NULL));
    int first_i = rank_row*N;
    int first_j = rank_col;
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i*n+j] = ((rank_row*n)+i)*N + (rank_col*n)+j+1;
            B[i*n+j] = ((rank_row*n)+i)*N + (rank_col*n)+j+1;
        }
    }
    
    double sum_C, total_sum_C;
    double start, end;

    // Time Cannon's Method/
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    cannon(n, A, B, &C);
    end = MPI_Wtime() - start;
    sum_C = mat_sum(n, C);
    MPI_Reduce(&sum_C, &total_sum_C, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("SumC %e\n", total_sum_C);
    MPI_Reduce(&end, &start, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Elapsed Time %e\n", start);
    free(C);


    free(A);
    free(B);

    MPI_Finalize();
    return 0;
}
