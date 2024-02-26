import numpy as np

from magpy.models import Model



def evolve_exact(model: Model, times, eigw_BZ, eigv_BZ, init_wavefunction):
    lattice_dims = init_wavefunction.shape[:-1]
    num_unit_cells = int(np.prod(lattice_dims))
    num_sublattices = init_wavefunction.shape[-1]

    init_wavefunction_flat = init_wavefunction \
        .reshape((num_unit_cells, num_sublattices))
    wavefunctions_flat = np.zeros(
        (len(times), num_unit_cells, num_sublattices), 
        dtype=np.complex128)
    bravais_coords_flat = \
        model.lattice.sample_Bravais_lattice_in_canonical_coords(lattice_dims) \
        .reshape((num_unit_cells, model.lattice.embedding_dim))
    momenta_BZ_flat = \
        model.lattice.reciprocal_lattice.sample_inverse_unit_cell(lattice_dims)\
        .reshape((num_unit_cells, model.lattice.embedding_dim))
    sigma_x = np.kron(np.eye(num_sublattices), np.array([[0, 1], [1, 0]]))
    sigma_z = np.kron(np.eye(num_sublattices), np.array([[1, 0], [0, -1]]))
    eigw_BZ_flat = eigw_BZ.reshape((num_unit_cells, 2*num_sublattices))
    eigv_BZ_flat = eigv_BZ.reshape((num_unit_cells, 2*num_sublattices, 2*num_sublattices))
    eigv_minus_BZ_flat = np.zeros(eigv_BZ_flat.shape, dtype=np.float64)
    eigv_minus_BZ_inv_flat = np.zeros(eigv_BZ_flat.shape, dtype=np.complex128)
    for k_idx in range(len(momenta_BZ_flat)):
        eigv_minus_BZ_flat[k_idx] = sigma_x @ eigv_BZ_flat[k_idx].conj() @ sigma_x
        eigv_minus_BZ_inv_flat[k_idx] = sigma_z @ eigv_minus_BZ_flat[k_idx].T.conj() @ sigma_z

    for t_idx, t in enumerate(times):
        for j in range(num_unit_cells):
            r_j = bravais_coords_flat[j]
            for s_ in range(num_sublattices):
                for i in range(num_unit_cells):
                    r_i = bravais_coords_flat[i]
                    for s in range(num_sublattices):
                        for k_idx, k in enumerate(momenta_BZ_flat):
                            for n in range(num_sublattices):
                                wavefunctions_flat[t_idx, j, s_] += init_wavefunction_flat[i, s] * np.exp(1j*k.dot(r_i - r_j) - 1j*eigw_BZ_flat[k_idx, 2*n]*t) * eigv_minus_BZ_flat[k_idx, 2*s+1, 2*n+1] * eigv_minus_BZ_inv_flat[k_idx, 2*n+1, 2*s_+1]

    wavefunctions = wavefunctions_flat.reshape((len(times), *lattice_dims, num_sublattices))
    return wavefunctions / num_unit_cells

# def to_LSWT_eigenspace(wavefunction, momenta_BZ_flat, bravais_coords_flat, eigv_minus_BZ_flat):
#     num_unit_cells, num_sublattices = bravais_coords_flat.shape[0:2]
#     wavefunction_eigenspace_flat = np.zeros(
#         (len(momenta_BZ_flat), num_sublattices), 
#         dtype=np.complex128)

#     for i in range(num_unit_cells):
#         for s in range(num_sublattices):
#             for k in range(len(momenta_BZ_flat)):
#                 for n in range(num_sublattices):
#                     wavefunction_eigenspace_flat[]

# def evolve_sequentially(LSWT_Hamiltonian_real_space, times):
    