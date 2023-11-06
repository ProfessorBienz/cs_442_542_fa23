#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>


void create_shared_mem(int proc_size)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);

    int ppn, node_rank;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &ppn);

    MPI_Win shared_window;
    int* shared_ptr;

    MPI_Info window_info;
    MPI_Info_create(&window_info);
    MPI_Info_set(window_info, "alloc_shared_noncontig", "true");

    MPI_Win_allocate_shared(proc_size*sizeof(int),
            sizeof(int),
            window_info,
            node_comm,
            &shared_ptr,
            &shared_window);

    MPI_Win_lock_all(0, shared_window);
    MPI_Win_sync(shared_window);
    MPI_Barrier(node_comm);

    MPI_Aint* sizes = new MPI_Aint[ppn];
    int* displs = new int[ppn];
    int** buf = new int*[ppn];

    for (int i = 0; i < ppn; i++)
    {
        MPI_Win_shared_query(shared_window, i, &(sizes[i]), &(displs[i]),
                &(buf[i]));
        if (node_rank == i) *buf[i] = i;
    }
    
    MPI_Win_unlock_all(shared_window);
    MPI_Barrier(node_comm);
    MPI_Win_flush_all(shared_window);

    for (int i = 0; i < ppn; i++)
    {
        printf("Rank %d, Sees Buf[%d] = %d\n", rank, i, *buf[i]);
    }

    delete[] sizes;
    delete[] displs;
    delete[] buf;

    MPI_Info_free(&window_info);
    MPI_Win_free(&shared_window);

    MPI_Comm_free(&node_comm);
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
 
    int n = atoi(argv[1]);
    create_shared_mem(n);

    return MPI_Finalize();
}
