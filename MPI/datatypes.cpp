#include "mpi.h"
#include "stdlib.h"
#include "stdio.h"

void extra_messages(int* mat, int* col, int n, int idx)
{
    int num_procs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Status status;
    for (int i = 0; i < n; i++)
    {
        if (rank == 0) MPI_Send(&(mat[i*n + idx]), 1, MPI_INT, 1, 1234, MPI_COMM_WORLD);
        else if (rank == 1) MPI_Recv(&(col[i]), 1, MPI_INT, 0,
                1234, MPI_COMM_WORLD, &status);
    }
}

void extra_copy(int* mat, int* col, int n, int idx)
{
    int num_procs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int* buffer = new int[n];
    MPI_Status status;
    for (int i = 0; i < n; i++)
    {
        buffer[i] = mat[i*n+idx];
    }

    if (rank == 0) MPI_Send(buffer, n, MPI_INT, 1, 1234, MPI_COMM_WORLD);
    else if (rank == 1) MPI_Recv(col, n, MPI_INT, 0, 1234, 
            MPI_COMM_WORLD, &status);

    delete[] buffer;
}

void datatype_col(int* mat, int* col, int n, int idx, MPI_Datatype datatype)
{
    int num_procs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Status status;

    if (rank == 0) MPI_Send(&(mat[idx]), 1, datatype, 1, 1234, MPI_COMM_WORLD);
    else if (rank == 1) MPI_Recv(col, n, MPI_INT, 0, 1234, 
        MPI_COMM_WORLD, &status);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = atoi(argv[1]);
    int print = 1;
    if (argc > 2)
        print = atoi(argv[2]);

    int* mat = new int[n*n];
    if (rank == 0)
        for (int i = 0; i < n*n; i++)
            mat[i] = i;
    int* col = new int[n];

    if (print && rank == 0)
    {
        printf("Initial Matrix : \n");
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                printf("%d ", mat[i*n+j]);
            printf("\n");
        }
    }

    double t0, tfinal;
    int n_iter = 100;

    int col_idx = 1;

    
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        extra_messages(mat, col, n, col_idx);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Extra Msg Cost: %e\n", t0);

    n_iter = 10000;


    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        extra_copy(mat, col, n, col_idx);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Extra Copy Cost: %e\n", t0);

    MPI_Datatype col_type;
    MPI_Type_vector(n, 1, n, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        datatype_col(mat, col, n, col_idx, col_type);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Col Datatype Cost: %e\n", t0);

    MPI_Type_free(&col_type);

    if (print && rank == 1)
    {
        printf("Received Column : \n");
        for (int i = 0; i < n; i++)
        {
            printf("%d\n", col[i]);
        }
    }

    delete[] mat;

    MPI_Finalize();
    return 0;
}
