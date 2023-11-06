#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <math.h>

void matmat(double* A, double* B, double* C, int first_n, int last_n, int n)
{
    double val;
    for (int i = first_n; i < last_n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            val = A[i*n+j];
            for (int k = 0; k < n; k++)
                C[i*n+k] += val * B[j*n+k];
        }
    }
}

void cannon(int n, double* A, double* B, double** C_ptr)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double* C = (double*)malloc(n*n*sizeof(double));
    double* A2 = (double*)malloc(n*n*sizeof(double));
    double* B2 = (double*)malloc(n*n*sizeof(double));
    double* A3 = (double*)malloc(n*n*sizeof(double));
    double* B3 = (double*)malloc(n*n*sizeof(double));


    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        double* send_A = A;
        double* send_B = B;
        double* recv_A = A2;
        double* recv_B = B2;
        double* tmp;
        #pragma omp barrier

        int sq_num_procs = sqrt(num_procs);
        int rank_row = rank / sq_num_procs;
        int rank_col = rank % sq_num_procs;

        int proc, shift;
        int proc_row, proc_col;
        int tag_a = 1234;
        int tag_b = 4321;
        int first_n, last_n;

        MPI_Request send_req_a, send_req_b, recv_req_a, recv_req_b;

        // Cannon Shift:
        #pragma omp master
        {

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

            MPI_Wait(&recv_req_a, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_req_b, MPI_STATUS_IGNORE);
            MPI_Wait(&send_req_a, MPI_STATUS_IGNORE);
            MPI_Wait(&send_req_b, MPI_STATUS_IGNORE);

            tag_a++;
            tag_b++;
            first_n = 0;
            last_n = 0;
        }
        
        if (tid > 0)
        {
            int n_active = omp_get_num_threads() - 1;
            int rank = tid - 1;
            int local_n = n / n_active;
            first_n = local_n * rank;
            int extra = n % n_active;
            if (rank < extra)
            {
                local_n++;
                first_n += rank;
            } 
            else first_n += extra;
            last_n = first_n + local_n;

            for (int i = first_n; i < last_n; i++) 
            {
                for (int j = 0; j < n; j++)
                {
                    C[i*n+j] = 0;
                }
            }
        }

        #pragma omp barrier

        recv_A = A3;
        recv_B = B3;
        send_A = A2;
        send_B = B2;


        int n_shifts = sq_num_procs - 1;
        for (int i = 0; i < n_shifts; i++)
        {
            #pragma omp master
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

                MPI_Wait(&recv_req_a, MPI_STATUS_IGNORE);
                MPI_Wait(&recv_req_b, MPI_STATUS_IGNORE);
                MPI_Wait(&send_req_a, MPI_STATUS_IGNORE);
                MPI_Wait(&send_req_b, MPI_STATUS_IGNORE);

                tag_a++;
                tag_b++;
            }
            matmat(send_A, send_B, C, first_n, last_n, n);

            #pragma omp barrier
        
            tmp = send_A;
            send_A = recv_A;
            recv_A = tmp;
            tmp = send_B;
            send_B = recv_B;
            recv_B = tmp;
            
        }
        matmat(send_A, send_B, C, first_n, last_n, n);
        #pragma omp barrier
    }

    free(A2);
    free(A3);
    free(B2);
    free(B3);

    *C_ptr = C;
}

double mat_sum(int n, double* C)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            sum += C[i*n+j];
        }
    }
    return sum;
}

int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    if (argc <= 1)
    {
        printf("Pass the matrix dimension (n for an nxn matrix) as a command line argument\n");
        return MPI_Finalize();
    }

    int N = atoi(argv[1]);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i*n+j] = ((rank_row*n)+i)*N + (rank_col*n)+j+1;
            B[i*n+j] = ((rank_row*n)+i)*N + (rank_col*n)+j+1;
        }
    }
    
    double sum_C, total_sum_C;
    double start, end;

    // Time Cannon's Method
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
