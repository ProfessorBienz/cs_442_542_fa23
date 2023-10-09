#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

void simple_test(int buf_size)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    MPI_File file;
    int* buf = (int*)malloc(buf_size*sizeof(int));
    MPI_File_open(MPI_COMM_WORLD, "test.out", 
            MPI_MODE_CREATE|MPI_MODE_WRONLY,
            MPI_INFO_NULL, &file);
    double t0 = MPI_Wtime();
    for (int i = 0; i < 1000; i++)
    {
        if (rank == 0) MPI_File_write_shared(file, buf, buf_size, MPI_INT, MPI_STATUS_IGNORE);
        //if (rank == 1) MPI_File_write_shared(file, buf, buf_size, MPI_INT, MPI_STATUS_IGNORE);
    }
    double tfinal = (MPI_Wtime() - t0) / 1000;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Time %e\n", t0);
    MPI_File_close(&file);

    free(buf);
}

void offset_test(int buf_size)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Status status;
    MPI_File file;
    MPI_Offset offset;
    int count;

    MPI_File_open(MPI_COMM_WORLD, "test_offset.out",
            MPI_MODE_CREATE|MPI_MODE_WRONLY,
            MPI_INFO_NULL, &file);

    int* buf = (int*)malloc(buf_size*sizeof(int));
    offset = rank * buf_size * sizeof(int);

    MPI_File_write_at(file, offset, buf, buf_size, MPI_INT, &status);
    MPI_Get_count(&status, MPI_INT, &count);
    printf("Rank %d wrote %d integers\n", rank, count);

    MPI_File_close(&file);

    free(buf);
}

void noncontig_test(int buf_size)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int* buf = (int*)malloc(buf_size*sizeof(int));
    MPI_Status status;
    MPI_Aint lb, extent;
    MPI_Datatype etype, filetype, contig;
    MPI_Offset displ;
    MPI_File file;
    int count;

    MPI_Type_contiguous(2, MPI_INT, &contig);
    lb = 0;
    extent = 2*num_procs*sizeof(int);
    MPI_Type_create_resized(contig, lb, extent, &filetype);
    MPI_Type_commit(&filetype);
    displ = 5*sizeof(int);
    etype = MPI_INT;

    MPI_File_open(MPI_COMM_WORLD, "test_noncontig.out",
            MPI_MODE_CREATE | MPI_MODE_WRONLY,
            MPI_INFO_NULL, &file);
    MPI_File_set_view(file, displ, etype, filetype, "native",
            MPI_INFO_NULL);

    MPI_File_write(file, buf, buf_size, MPI_INT, &status);
    MPI_Get_count(&status, MPI_INT, &count);
    printf("Rank %d wrote %d integers\n", rank, count);

    MPI_File_close(&file);

    free(buf);
}

void write_mat()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = 10000;
    double** A = (double**)malloc(n*sizeof(double*));
    for (int i = 0; i < n; i++)
        A[i] = (double*)malloc(n*sizeof(double));

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "A.out",
            MPI_MODE_CREATE|MPI_MODE_WRONLY,
            MPI_INFO_NULL, &file);
    if (rank == 0) 
        for (int i = 0; i < n; i++)
            MPI_File_write(file, A[i], n, MPI_DOUBLE, MPI_STATUS_IGNORE);

    for (int i = 0; i < n; i++)
        free(A[i]);
    free(A);
    
    MPI_File_close(&file);
}


void seek_read(double* A, int n, int local_n, int first_n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "A.out",
            MPI_MODE_RDONLY,
            MPI_INFO_NULL, &file);

    MPI_Offset offset = n * num_procs * sizeof(double);
    MPI_Offset first_offset = n * rank * sizeof(double);
    MPI_File_seek(file, first_offset, MPI_SEEK_SET);
    for (int i = 0; i < local_n; i++)
    {
        MPI_File_read(file, &(A[i*n]), n, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_seek(file, offset, MPI_SEEK_CUR);
    }

    MPI_File_close(&file);
}

void seek_read_all(double* A, int n, int local_n, int first_n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "A.out",
            MPI_MODE_RDONLY,
            MPI_INFO_NULL, &file);

    MPI_Offset offset = n * num_procs * sizeof(double);
    MPI_Offset first_offset = n * rank * sizeof(double);
    MPI_File_seek(file, first_offset, MPI_SEEK_SET);
    for (int i = 0; i < local_n; i++)
    {
        MPI_File_read_all(file, &(A[i*n]), n, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_seek(file, offset, MPI_SEEK_CUR);
    }

    MPI_File_close(&file);
}

void read_at(double* A, int n, int local_n, int first_n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "A.out",
            MPI_MODE_RDONLY,
            MPI_INFO_NULL, &file);

    MPI_Offset offset;
    for (int i = 0; i < local_n; i++)
    {
        offset = (n* (rank + num_procs*i)) * sizeof(double);
        MPI_File_read_at(file, offset, &(A[i*n]), n, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&file);
}

void read_at_all(double* A, int n, int local_n, int first_n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "A.out",
            MPI_MODE_RDONLY,
            MPI_INFO_NULL, &file);

    MPI_Offset offset;
    for (int i = 0; i < local_n; i++)
    {
        offset = (n* (rank + num_procs*i)) * sizeof(double);
        MPI_File_read_at_all(file, offset, &(A[i*n]), n, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&file);
}


void view_read(double* A, int n, int local_n, int first_n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Status status;
    int count;

    MPI_Aint lb, extent;
    MPI_Datatype filetype, contig;
    MPI_Offset displ;

    MPI_Type_contiguous(n, MPI_DOUBLE, &contig);
    lb = 0;
    extent = n*num_procs*sizeof(double);
    MPI_Type_create_resized(contig, lb, extent, &filetype);
    MPI_Type_commit(&filetype);
    displ = n*rank*sizeof(double);
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "A.out",
            MPI_MODE_RDONLY,
            MPI_INFO_NULL, &file);

    MPI_File_set_view(file, displ, MPI_DOUBLE, filetype, "native",
            MPI_INFO_NULL);

    MPI_File_read(file, A, local_n*n, MPI_DOUBLE , &status);
    MPI_Get_count(&status, MPI_DOUBLE, &count);

    MPI_Type_free(&contig);
    MPI_Type_free(&filetype);

    MPI_File_close(&file);

}

void view_read_all(double* A, int n, int local_n, int first_n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPI_Status status;
    int count;

    MPI_Aint lb, extent;
    MPI_Datatype filetype, contig;
    MPI_Offset displ;

    MPI_Type_contiguous(n, MPI_DOUBLE, &contig);
    lb = 0;
    extent = n*num_procs*sizeof(double);
    MPI_Type_create_resized(contig, lb, extent, &filetype);
    MPI_Type_commit(&filetype);
    displ = n*rank*sizeof(double);
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "A.out",
            MPI_MODE_RDONLY,
            MPI_INFO_NULL, &file);

    MPI_File_set_view(file, displ, MPI_DOUBLE, filetype, "native",
            MPI_INFO_NULL);

    MPI_File_read_all(file, A, local_n*n, MPI_DOUBLE , &status);
    MPI_Get_count(&status, MPI_DOUBLE, &count);

    MPI_Type_free(&contig);
    MPI_Type_free(&filetype);

    MPI_File_close(&file);

}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;

    //offset_test(2);
    
    int n = 10000; 
    int local_n = n / num_procs;
    int extra = n % num_procs;
    int first = local_n * rank;
    if (rank < extra) 
    {
        local_n++;
        first += rank;
    }
    else
    {
        first += extra;
    }
    double* A = (double*)malloc(n*local_n*sizeof(double*));

    int n_iter = 1000;

    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        seek_read(A, n, local_n, first);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Seek and Read Time %e\n", t0);

    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        seek_read_all(A, n, local_n, first);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Seek and ReadAll Time %e\n", t0);

    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        read_at(A, n, local_n, first);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Read At Time %e\n", t0);

    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        read_at_all(A, n, local_n, first);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Read At All Time %e\n", t0);


    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        view_read(A, n, local_n, first);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("View and Read Time %e\n", t0);

    t0 = MPI_Wtime();
    for (int i = 0; i < n_iter; i++)
        view_read_all(A, n, local_n, first);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("View and ReadAll Time %e\n", t0);

    free(A);

    return MPI_Finalize();
}
