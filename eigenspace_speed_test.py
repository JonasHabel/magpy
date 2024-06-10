import numpy as np
from magpy.largeS.eigenspace import compute_magnon_Hamiltonian
import timeit

np.random.seed(1)

eigvs = [*np.random.rand(4, 12, 12).astype(complex)]
magnon_H_mom_space = np.random.rand(12, 12, 12, 12).astype(complex)
repeats = 10


def alternative_einsum(eigvs, magnon_H_mom_space):
    order = len(eigvs)

    einsum_str = "".join([chr(97 + i) for i in range(order)])
    if order >= 1:
        einsum_str += ","
    einsum_str += ",".join([chr(97 + i) + chr(65 + i) for i in range(order)])
    einsum_str += "->"
    einsum_str += "".join([chr(65 + i) for i in range(order)])
    print(einsum_str)

    return np.einsum(einsum_str, magnon_H_mom_space, *eigvs)


def tensordot(eigvs, magnon_H_mom_space):
    order = len(eigvs)
    magnon_H_eigenspace = magnon_H_mom_space.copy()

    for n, eigv in enumerate(eigvs):
        magnon_H_eigenspace = np.tensordot(magnon_H_eigenspace, eigv, axes=[[n], [0]])
        magnon_H_eigenspace = np.moveaxis(magnon_H_eigenspace, -1, n)
    
    return magnon_H_eigenspace


def profile(func):
    print(timeit.repeat(
        f"{func}(eigvs, magnon_H_mom_space)",
        setup="",
        globals=globals(), repeat=repeats, number=1))


#profile("compute_magnon_Hamiltonian")
#profile("alternative_einsum")
profile("tensordot")


eigvs = [*np.random.rand(1, 12, 12).astype(complex)]
magnon_H_mom_space = np.random.rand(12).astype(complex)

print(np.allclose(alternative_einsum(eigvs, magnon_H_mom_space), compute_magnon_Hamiltonian(eigvs, magnon_H_mom_space)))
print(np.allclose(tensordot(eigvs, magnon_H_mom_space), compute_magnon_Hamiltonian(eigvs, magnon_H_mom_space)))
