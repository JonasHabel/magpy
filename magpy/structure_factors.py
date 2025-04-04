import numpy as np
from magpy.models import Model


"""
magnon_greens_functions = [
    [<a a^†>], [<a a>],
    [<a^† a^†>], [<a^† a],
]

"""
def compute_structure_factor(magnon_greens_functions, momentum, eigvs, model: Model):
    assert len(magnon_greens_functions.shape) == 3
    assert len(eigvs.shape) == 2
    assert magnon_greens_functions.shape[1] % 2 == 0

    num_freqs = magnon_greens_functions.shape[0]
    num_bands = magnon_greens_functions.shape[1] // 2

    assert magnon_greens_functions.shape[2] == 2*num_bands
    assert eigvs.shape[0] == eigvs.shape[1] == 2*num_bands

    sigma_z_BdG = np.kron(np.eye(num_bands), np.array([[1, 0], [0, -1]]))
    eigvs_conj = eigvs.T.conj()
    Hadamard = np.kron(np.eye(num_bands), np.array([[1, 1], [-1j, 1j]]))
    GF_eigenspace_basis = magnon_greens_functions
    GF_sublattice_basis = \
        np.einsum("sn,...nm,mt->...st", eigvs, GF_eigenspace_basis, eigvs_conj)
    GF_spin_basis = \
        np.einsum("as,...st,tb->...ab", Hadamard, GF_sublattice_basis, Hadamard.T.conj())

    S = model.get_onsite_spin_quantum_numbers()
    sublattice_offsets = model.lattice.sublattices
    u = np.sqrt(S) * np.exp(1j*sublattice_offsets.dot(momentum)) # the sublattice structure factor
    u_matrix = np.kron(np.outer(u.conj(), u), np.eye(2))

    # S_{αβ}(ω)   (α,β = x,y)
    structure_factor = np.zeros((num_freqs, 2, 2), dtype=np.complex128)

    for n, GF_for_freq in enumerate(GF_spin_basis):
        # TODO prefactor??
        GF_antiherm = 0.5j * (GF_for_freq - GF_for_freq.T.conj())
        # reshaped axes: (sublattice_i, spin-dir_i, sublattice_j, spin-dir_j)
        struct_fact_sublattice = \
            (u_matrix * GF_antiherm).reshape((num_bands, 2, num_bands, 2))
        structure_factor[n] = np.sum(
            struct_fact_sublattice,
            axis=(0, 2),
        )   # sum over sublattices

    return structure_factor
