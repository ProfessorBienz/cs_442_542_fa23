from mpi4py import MPI
import numpy as np

Comm = MPI.COMM_WORLD

num_procs = Comm.Get_size()
rank = Comm.Get_rank()

root = 0
array = ""
if rank == root:
    array = ["hello", 45, [1234], "goodbye"]

array = Comm.scatter(array, root)

print(array)

