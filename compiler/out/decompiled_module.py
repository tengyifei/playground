import jax
from jax.numpy import *
from jax.experimental import sparse
from jax._src import prng
from mpi4py import MPI
def f(Var(id=135574744324672):float32[4,8], Var(id=135574744327040):float32[8,4]):
    Var(id=135574744329920):float32[4,4] = tensordot(Var(id=135574744324672):float32[4,8], Var(id=135574744327040):float32[8,4],axes=((1,), (0,)))
    Var(id=135574744328192):float32[4,4] = sin(Var(id=135574744329920):float32[4,4])
    return Var(id=135574744328192):float32[4,4]
