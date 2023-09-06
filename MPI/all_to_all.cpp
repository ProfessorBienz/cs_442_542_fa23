#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <vector>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Declare variable I want to send
    int size = 10000; // Each message is 10000 ints
    std::vector<int> array(size*num_procs);
    std::vector<int> array_recv(size);

    // Allocate arrays of MPI_Request (one per send, one per recv)
    std::vector<MPI_Request> send_req(num_procs);
    std::vector<MPI_Request> recv_req(num_procs);

    // Start non-blocking sends and recvs for each msg
    for (int i = 0; i < num_procs; i++)
    {
        MPI_Isend(&(array[size*i]), size, MPI_INT, i, 1234, 
                MPI_COMM_WORLD, &(send_req[i]));
        MPI_Irecv(array_recv.data(), size, MPI_INT, i, 1234, 
                MPI_COMM_WORLD, &(recv_req[i]));
    }

    // Wait for all requests in an array to complete (.data() gives pointer to data)
    MPI_Waitall(num_procs, send_req.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(num_procs, recv_req.data(), MPI_STATUSES_IGNORE);

    MPI_Finalize();
    return 0;
}
