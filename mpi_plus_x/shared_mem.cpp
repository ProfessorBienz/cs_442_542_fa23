#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    // Get MPI information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Check that command line argument is present
    if (argc < 2)
    {
        if (rank == 0) printf("Pass window size `n` as a command line argument!\n");
        return MPI_Finalize();
    }
 
    // Get per-process vector size from command line
    int proc_size = atoi(argv[1]);

    // Split MPI_COMM_WORLD into new communicators, one for each shared memory region
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
            MPI_INFO_NULL, &node_comm);

    // Get node-wise MPI information (rank within the node, number of processes per node)
    int ppn, node_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ppn);

    // Create shared window
    int* shared_mem_ptr = NULL;
    MPI_Win shared_mem_win;
    MPI_Info win_info;
    MPI_Info_create(&win_info);
    //MPI_Info_set(win_info, "alloc_shared_noncontig", "true");
    MPI_Win_allocate_shared(proc_size*sizeof(int), sizeof(int), win_info, node_comm, 
            &shared_mem_ptr, &shared_mem_win);

    // Lock all members of the window so that only one process can operate on window object at a time
    MPI_Win_lock_all(0, shared_mem_win);

    // Synchronize private and public copies in the window
    MPI_Win_sync(shared_mem_win);

    // Wait for all processes to synchronize (lock-all is not blocking)
    MPI_Barrier(node_comm);

    // Allocates three arrays
    MPI_Aint* sizes = new MPI_Aint[ppn];
    int* displs = new int[ppn];
    int** buf = new int*[ppn];

    for (int i = 0; i < ppn; i++)
    {
        // Queries local address for remote memory segments of shared_mem_ptr 
        // Returns:
        //   - size of shared_mem_ptr on process i
        //   - local unit size (number of bytes per datatype) 
        //   - pointer to buffer, which can be used to load/store data
        MPI_Win_shared_query(shared_mem_win, i, &(sizes[i]), &(displs[i]), &(buf[i]));

        // All processes can now access buffer (which contains the shared_mem_ptr allocated on each process)
        if ((node_rank+1)%ppn == i)
        {
            int n_elements = sizes[i] / displs[i];
            for (int j = 0; j < n_elements; j++)
            {
                printf("Address of buf[%d][%d] = %p\n", i, j, &(buf[i][j]));
                buf[i][j] = i;
                printf("i = %d, j = %d: Rank %d Modified Buf[%d][%d] = %d\n", i, j, node_rank, i, j, buf[i][j]);
            }
        }
    }

    MPI_Win_flush_all(shared_mem_win);
    MPI_Win_unlock_all(shared_mem_win);
    MPI_Barrier(node_comm);

    if (rank == 0)
    {
        for (int i = 0; i < ppn; i++)
        {
            int n_elements = sizes[i] / displs[i];
            printf("Process %d's buffer size %lu, displs %d, n_elements %d\n", i, sizes[i], displs[i], n_elements);
            for (int j = 0; j < n_elements; j++)
            {
                printf("Buf[%d][%d] = %d\n", i, j, buf[i][j]);
            }
        }
    }


    delete[] sizes;
    delete[] displs;
    delete[] buf;

    MPI_Barrier(node_comm);
    MPI_Info_free(&win_info);
    MPI_Win_free(&shared_mem_win);
    MPI_Comm_free(&node_comm);



    return MPI_Finalize();
}
