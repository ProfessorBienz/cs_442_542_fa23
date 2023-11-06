#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, num_procs;

#pragma omp parallel shared(rank, num_procs)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        int thread = omp_get_thread_num();
        int n_threads = omp_get_num_threads();

        printf("Hello World from thread %d of %d spawned by rank %d of %d\n", thread, n_threads, rank, num_procs);
    }

    MPI_Finalize();
    return 0;
}
