#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <unistd.h>

int check_error(int ierr)
{
    int errclass;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (ierr == MPI_SUCCESS)
        return 0;

    MPI_Error_class(ierr, &errclass);
    if (errclass == MPI_ERR_RANK)
        printf("Error! Invalid rank in method called by rank %d\n", rank);
    else if (errclass == MPI_ERR_BUFFER)
        printf("Error! Invalid buffer pointer in method called by rank %d\n", rank);
    else if (errclass == MPI_ERR_COUNT)
        printf("Error! Invalid count argument in method called by rank %d\n", rank);
    else if (errclass == MPI_ERR_TYPE)
        printf("Error! Invalid datatype argument in method called by rank %d\n", rank);
    else if (errclass == MPI_ERR_TAG)
        printf("Error! Invalid tag argument in method called by rank %d\n", rank);
    else if (errclass == MPI_ERR_COMM)
        printf("Error! Invalid communicator argument in method called by rank %d\n", rank);
    else if (errclass == MPI_ERR_REQUEST)
        printf("Error! Invalid request argument in method called by rank %d\n", rank);
    else if (errclass == MPI_ERR_ROOT)
        printf("Error! Invalid root argument in method called by rank %d\n", rank);
    else if (errclass == MPI_ERR_GROUP)
        printf("Error! Invalid group argument in method called by rank %d\n", rank);

    return 1;
}

void error0(int n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int tag = 0;
    double* sendbuf = new double[n];
    double* recvbuf = new double[n];
    MPI_Status status;

    MPI_Send(sendbuf, n, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD);
    MPI_Recv(recvbuf, n, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);

    delete[] sendbuf;
    delete[] recvbuf;
}

void error1(int n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int tag = 0;
    double* sendbuf = new double[n];
    double* recvbuf = new double[n];
    MPI_Status status;
    if (rank % 2 == 0)
    {
        MPI_Send(sendbuf, n, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD);
        MPI_Recv(recvbuf, n, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);
    }
    else
    {
        MPI_Send(sendbuf, n, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD);
        MPI_Recv(recvbuf, n, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD, &status);
    }

    delete[] sendbuf;
    delete[] recvbuf;

}

void error2(int n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int tag = 0;
    int* sendbuf_int = new int[n];
    int* recvbuf_int = new int[n];
    double* sendbuf_dbl = new double[n];
    double* recvbuf_dbl = new double[n];
    MPI_Status status;

    if (rank % 2 == 0)
    {
        MPI_Send(sendbuf_int, n, MPI_INT, rank+1, tag, MPI_COMM_WORLD);
        MPI_Send(sendbuf_dbl, n, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD);
        MPI_Recv(recvbuf_int, n, MPI_INT, rank+1, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(recvbuf_dbl, n, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);
    }
    else
    {
        MPI_Recv(recvbuf_dbl, n, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(recvbuf_int, n, MPI_INT, rank-1, tag, MPI_COMM_WORLD, &status);
        MPI_Send(sendbuf_dbl, n, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD);
        MPI_Send(sendbuf_int, n, MPI_INT, rank-1, tag, MPI_COMM_WORLD);
    }

    delete[] sendbuf_int;
    delete[] sendbuf_dbl;
    delete[] recvbuf_int;
    delete[] recvbuf_dbl;
}

void error3(int n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int tag = 0;
    double* sendbuf = new double[n];
    double* recvbuf = new double[n];
    MPI_Status status;
    int ierr;

    ierr = MPI_Gather(sendbuf, n, MPI_DOUBLE, recvbuf, n, MPI_DOUBLE, 4, MPI_COMM_WORLD);

    check_error(ierr);

    delete[] sendbuf;
    delete[] recvbuf;
}

void error4(int n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int tag = 0;
    MPI_Status status;
    
    int* data = new int[n*num_procs];
    for (int i = 0; i < n*num_procs; i++)
        data[i] = num_procs;

    std::vector<int> send_buffer;
    std::vector<int> recv_buffer(n*num_procs);
    std::vector<MPI_Request> send_requests(num_procs);
    std::vector<MPI_Request> recv_requests(num_procs);

    int* address = send_buffer.data();
    for (int i = 0; i < num_procs; i++)
    {
        for (int j = 0; j < n; j++)
        {
            send_buffer.push_back(data[j*num_procs+i]);
        }
        if (send_buffer.data() != address)
            printf("SendBuffer address changed!\n");
        MPI_Isend(&(send_buffer[i*n]), n, MPI_INT, i, tag, MPI_COMM_WORLD, &(send_requests[i]));
    }

    for (int i = 0; i < num_procs; i++)
    {
        MPI_Irecv(&(recv_buffer[i*n]), n, MPI_INT, i, tag, MPI_COMM_WORLD, &(recv_requests[i]));
    }

    MPI_Waitall(num_procs, send_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(num_procs, recv_requests.data(), MPI_STATUSES_IGNORE);

    for (int i = 0; i < n*num_procs; i++)
        if (recv_buffer[i] != num_procs)
            printf("Rank %d, Recv Buffer[%d] = %d (not %d)!\n", rank, i, recv_buffer[i], num_procs);

    delete[] data;

}

void error5(int n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int tag = 0;
    double* sendbuf0 = new double[n];
    double* sendbuf1 = new double[n+1];
    double* recvbuf0 = new double[n];
    double* recvbuf1 = new double[n+1];
    MPI_Status status;
    MPI_Request req[2];
    if (rank % 2 == 0)
    {
        MPI_Isend(sendbuf0, n, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &(req[0]));
        MPI_Isend(sendbuf1, n+1, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &(req[1]));
    }
    else
    {
        MPI_Irecv(recvbuf1, n+1, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD, &(req[0]));
        MPI_Irecv(recvbuf0, n, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD, &(req[1]));
    }

    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

    delete[] sendbuf0;
    delete[] sendbuf1;
    delete[] recvbuf0;
    delete[] recvbuf1;

}

void logical_bug(int n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int N = n;
    n /= num_procs;
    std::vector<double> x(n, 0.5);
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i]*x[i];
    int total_sum = 0;
    MPI_Reduce(&sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Sum %d\n", total_sum);
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc <= 1)
    {
        if (rank == 0) printf("Add size to command line\n");
        MPI_Finalize();
        return 0;
    }

    int n = atoi(argv[1]);

    //MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
    
    error4(n);
    

    MPI_Finalize();
    return 0;
}
