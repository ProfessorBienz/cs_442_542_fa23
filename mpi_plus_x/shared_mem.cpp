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
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ppn);

    int node_size = proc_size * ppn;

    int* shared_mem_ptr = NULL;
    MPI_Win shared_mem_win;
    MPI_Info win_info;
    MPI_Info_create(&win_info);
    MPI_Info_set(win_info, "alloc_shared_noncontig", "true");
    //MPI_Info_set(win_info, "alloc_shared_noncontig", "false");
    MPI_Win_allocate_shared(sizeof(int), sizeof(int), win_info, node_comm, 
            &shared_mem_ptr, &shared_mem_win);

    MPI_Win_lock_all(0, shared_mem_win);

    MPI_Win_sync(shared_mem_win);
    MPI_Barrier(node_comm);

    MPI_Aint* sizes = new MPI_Aint[ppn];
    int* displs = new int[ppn];
    int** buf = new int*[ppn];


    for (int i = 0; i < ppn; i++)
    {
        MPI_Win_shared_query(shared_mem_win, i, &(sizes[i]), &(displs[i]), &(buf[i]));
        if (node_rank == 3)
        {
            *buf[i] = node_rank + 100;
            printf("Rank %d Modified Buf[%d] = %d\n", node_rank, i, *buf[i]);
        }
        else printf("Rank %d is just watching change\n", node_rank);
    }

    MPI_Win_unlock_all(shared_mem_win);
    MPI_Barrier(node_comm);

    for (int i = 0; i < ppn; i++)
    {
        if (buf[i] != NULL)
            printf("Rank %d, Target %d, Buf %d, Size %d, Displs %d\n", 
                    node_rank, i, *buf[i], (int)sizes[i], displs[i]);
    }


    delete[] sizes;
    delete[] displs;
    delete[] buf;

    MPI_Win_free(&shared_mem_win);
    MPI_Comm_free(&node_comm);
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
 
    int n = atoi(argv[1]);
    create_shared_mem(n);

    return MPI_Finalize();
}
