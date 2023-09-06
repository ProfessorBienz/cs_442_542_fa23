from mpi4py import MPI
import numpy as np

Comm = MPI.COMM_WORLD

size = Comm.Get_size()
rank = Comm.Get_rank()

proc = rank - 1
if (rank % 2 == 0):
    proc = rank + 1

size = 10000
array = np.random.rand(size);
array_recv = np.zeros(size);

comm_type = 0

if comm_type == 0:
    if (rank % 2):
        Comm.Send([array, MPI.DOUBLE], proc, tag=1234)
        Comm.Recv(array_recv, proc, tag=2345)
    else:
        Comm.Recv(array_recv, proc, tag=1234)
        Comm.Send(array, proc, tag=2345)

elif comm_type == 1:
    Comm.Sendrecv(array, proc, sendtag=1234, recvbuf=array_recv, source=proc, recvtag=1234)

elif comm_type == 2:
    send_req = Comm.Isend(array, proc, tag=1234)
    recv_req = Comm.Irecv(array_recv, proc, tag=1234)
    send_req.Wait()
    recv_req.Wait()

print(array_recv)

