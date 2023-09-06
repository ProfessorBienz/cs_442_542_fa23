from mpi4py import MPI

Comm = MPI.COMM_WORLD

size = Comm.Get_size()
rank = Comm.Get_rank()

print("Hello World from Rank %d of %d\n"%(rank, size))
