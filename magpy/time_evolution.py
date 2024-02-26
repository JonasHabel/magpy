import numpy as np
from numba import njit
from magpy.lattice import BravaisLattice
from magpy.models import Model



def evolve_exact(model: Model, times, eigw_BZ, eigv_BZ, init_wavefunction):
    lattice_dims = init_wavefunction.shape[:-1]
    num_unit_cells = int(np.prod(lattice_dims))
    num_sublattices = init_wavefunction.shape[-1]

    bravais_coords_flat = model.lattice \
        .sample_Bravais_lattice_in_canonical_coords(lattice_dims) \
        .reshape((num_unit_cells, model.lattice.embedding_dim))
    momenta_BZ_flat = model.lattice.reciprocal_lattice \
        .sample_inverse_unit_cell(lattice_dims, as_meshgrid=False)\
        .reshape((num_unit_cells, model.lattice.embedding_dim))

    wavefunctions_flat = evolve_exact_jit(times, eigw_BZ, eigv_BZ, momenta_BZ_flat, bravais_coords_flat, init_wavefunction, num_unit_cells)
    wavefunctions = wavefunctions_flat.reshape((len(times), *lattice_dims, num_sublattices))
    return wavefunctions


#@njit
def evolve_exact_jit(times, eigw_BZ, eigv_BZ, momenta_BZ_flat, bravais_coords_flat, init_wavefunction, num_unit_cells):
    num_sublattices = init_wavefunction.shape[-1]

    sigma_x = np.kron(np.eye(num_sublattices).astype(np.complex128), np.array([[0, 1], [1, 0]], dtype=np.complex128))
    sigma_z = np.kron(np.eye(num_sublattices).astype(np.complex128), np.array([[1, 0], [0, -1]], dtype=np.complex128))
    pos_eigw_BZ_flat = eigw_BZ.reshape((num_unit_cells, 2*num_sublattices))[:,::2]
    eigv_BZ_flat = eigv_BZ.reshape((num_unit_cells, 2*num_sublattices, 2*num_sublattices))
    eigv_minus_BZ_flat = np.zeros(eigv_BZ_flat.shape, dtype=np.complex128)
    eigv_minus_BZ_inv_flat = np.zeros(eigv_BZ_flat.shape, dtype=np.complex128)
    for k_idx in range(len(momenta_BZ_flat)):
        eigv_minus_BZ_flat[k_idx] = sigma_x @ eigv_BZ_flat[k_idx].conj() @ sigma_x
        eigv_minus_BZ_inv_flat[k_idx] = sigma_z @ eigv_minus_BZ_flat[k_idx].T.conj() @ sigma_z
    creator_eigv_minus_BZ_flat = eigv_minus_BZ_flat[:, 1::2, 1::2]
    creator_eigv_minus_BZ_inv_flat = eigv_minus_BZ_inv_flat[:, 1::2, 1::2]

    init_wavefunction_flat = init_wavefunction \
        .reshape((num_unit_cells, num_sublattices)) \
        .astype(np.complex128)
    wavefunctions_flat = np.zeros(
        (len(times), num_unit_cells, num_sublattices), 
        dtype=np.complex128)

    for k_idx, k in enumerate(momenta_BZ_flat):
        fourier_phase_matrix = np.zeros((num_unit_cells, num_unit_cells), dtype=np.complex128)
        for j in range(num_unit_cells):
            r_j = bravais_coords_flat[j]
            for i in range(num_unit_cells):
                r_i = bravais_coords_flat[i]
                fourier_phase_matrix[j, i] = np.exp(1j*k.dot(r_i - r_j))

        for t_idx, t in enumerate(times):
            time_evol_phase_matrix = np.diag(np.exp(-1j*pos_eigw_BZ_flat[k_idx]*t))
            time_evol_sublattice_space_matrix = (creator_eigv_minus_BZ_flat[k_idx] @ time_evol_phase_matrix @ creator_eigv_minus_BZ_inv_flat[k_idx])
            wavefunctions_flat[t_idx] += fourier_phase_matrix @ init_wavefunction_flat @ time_evol_sublattice_space_matrix

    return wavefunctions_flat / num_unit_cells



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
    



def get_Gaussian_wave_packet(lattice: BravaisLattice, lattice_dims, init_pos, init_mom):
    num_unit_cells = int(np.prod(lattice_dims))
    num_sublattices = lattice.num_sites_unit_cell
    num_sites_total = num_unit_cells * num_sublattices
    all_site_positions_flat = lattice \
        .sample_full_lattice_in_canonical_coords(lattice_dims) \
        .reshape((num_unit_cells, num_sublattices, lattice.embedding_dim))

    def f(site_pos):
        return np.exp(-0.5*np.sum((site_pos - init_pos)**2) - 1j*init_mom.dot(site_pos))

    wave_packet_flat = np.zeros(
        (num_unit_cells, num_sublattices), 
        dtype=np.complex128)
    
    for site_idx, site_pos_unit_cell in enumerate(all_site_positions_flat):
        for subl_idx, site_pos in enumerate(site_pos_unit_cell):
            wave_packet_flat[site_idx, subl_idx] = f(site_pos)

    return wave_packet_flat.reshape((*lattice_dims, num_sublattices))


def get_expectation_values(wavefunctions, observable, lattice: BravaisLattice):
    num_times = wavefunctions.shape[0]
    lattice_dims = wavefunctions.shape[1:-1]
    num_unit_cells = int(np.prod(lattice_dims))
    num_sublattices = wavefunctions.shape[-1]
    num_sites_total = num_unit_cells * num_sublattices

    wavefunctions_transformed = np.array([
        observable(wavefunction_at_time, lattice) for wavefunction_at_time in wavefunctions
    ])
    observable_dims = wavefunctions_transformed.shape[1:-lattice.embedding_dim-1]
    wavefunctions_flat = wavefunctions \
        .reshape((num_times, num_sites_total))
    wavefunctions_transformed_flat = wavefunctions_transformed \
        .reshape((num_times, int(np.prod(observable_dims)), num_sites_total))

    expectation_value_flat = np.einsum("tx,tdx->td", 
        wavefunctions_flat.conj(), 
        wavefunctions_transformed_flat)
    
    return expectation_value_flat.reshape((num_times, *observable_dims))


class observables:
    @staticmethod
    def position(wavefunction, lattice: BravaisLattice):
        lattice_dims = wavefunction.shape[:-1]
        coords = lattice.sample_full_lattice_in_canonical_coords(lattice_dims)
        coords_meshgrid = \
            np.transpose(coords, axes=[-1, *range(len(coords.shape)-1)]) \
              .astype(np.complex128)
        
        for d in range(lattice.embedding_dim):
            coords_meshgrid[d] *= wavefunction

        return coords_meshgrid
