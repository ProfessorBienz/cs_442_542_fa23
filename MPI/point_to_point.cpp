#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    // Declare variable I want to send
    int size[10000];
    int size_recv[10000]; // separate variable for recvs 
                          
    // How many ints from size to actually send/recv 
    // Should be between 1 and 10000
    int n = 10000; // Up to 8k bytes, sends will not hang

    int proc = rank + 1;
    if (rank%2 ==1 ) proc = rank - 1;

    // How to send messages
    // msg_type : 0 send and recv
    //            1 sendrecv
    //            2 isend, irecv, and wait
    int msg_type = 0;

    MPI_Request send_req, recv_req;

    if (msg_type == 0) // Send and Recv
    {
        if (rank % 2 == 0) // Even processes send first
        {
            MPI_Send(size, n, MPI_INT, proc, 
                    1234, MPI_COMM_WORLD);
            MPI_Recv(size, n, MPI_INT, proc, 
                    2341, MPI_COMM_WORLD, &status);
        }
        else // Odd processes recv first
        {
            MPI_Recv(size, n, MPI_INT, proc, 
                    1234, MPI_COMM_WORLD, &status);
            MPI_Send(size, n, MPI_INT, proc, 
                    2341, MPI_COMM_WORLD);
        }
    }
    else if (msg_type == 1) // Sendrecv
    {
        MPI_Sendrecv(size, n, MPI_INT, proc, 1234, 
                size_recv, n, MPI_INT, proc, 1234, 
                MPI_COMM_WORLD, &status);
    }
    else if (msg_type == 2) // Nonblocking Isend/Irecv/Wait
    {
        MPI_Isend(size, n, MPI_INT, proc, 1234, 
                MPI_COMM_WORLD, &send_req);
        MPI_Irecv(size, n, MPI_INT, proc, 1234, 
                MPI_COMM_WORLD, &recv_req);

        MPI_Wait(&send_req, &status);
        MPI_Wait(&recv_req, &status);
    }

    MPI_Finalize();
    return 0;
}
