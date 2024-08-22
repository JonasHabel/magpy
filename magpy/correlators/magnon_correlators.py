import numpy as np
from magpy.momenta_utils import CollapseMomenta, RestoreMomenta, Target


@CollapseMomenta(
    targets=(
        Target(arg_idx=0, first_momentum_idx=0, is_tensor=False), # ks_BZ
        Target(arg_idx=1, first_momentum_idx=0, is_tensor=False), # eigvs_BZ
    )
)
def compute_real_space_correlator_LSWT(
    ks_BZ, eigvs_BZ, bravais_coords,
):
    assert len(ks_BZ) == 1
    assert len(eigvs_BZ) == 1
    ks_BZ, eigvs_BZ = ks_BZ[0], eigvs_BZ[0]

    dims_real_space = bravais_coords.shape[:-1]
    num_bravais_coords = int(np.prod(dims_real_space))
    num_ks_BZ = len(ks_BZ)
    subl_shape = eigvs_BZ.shape[-2:]
    bravais_coords_flat = bravais_coords.reshape((num_bravais_coords,))
    correlators_real_space = np.zeros(
        (num_bravais_coords, *subl_shape),
        dtype=np.complex128,
    )

    for k, eigvs_for_k in zip(ks_BZ, eigvs_BZ):
        correlators_mom_space = \
            compute_momentum_space_correlator_LSWT(eigvs_for_k)
        
        for nx, x in enumerate(bravais_coords_flat):
            correlators_real_space[nx] += \
                np.exp(-1j*k.dot(x)) * correlators_mom_space
        
    correlators_real_space /= num_ks_BZ   # Fourier trafo normalization

    return correlators_real_space.reshape((*dims_real_space, *subl_shape))
        


def compute_momentum_space_correlator_LSWT(eigvs_k):
    # ensure right shape (sublattice idx, band idx)
    assert len(eigvs_k.shape) == 2   
    # ensure equal number of particle and hole states
    assert np.all(np.array(eigvs_k.shape) % 2 == 0)

    num_bands = len(eigvs_k) // 2
    sigma_z_ph = np.kron(np.eye(num_bands), np.diag([1, -1]))
    sigma_x_ph = np.kron(np.eye(num_bands), np.array([[0, 1], [1, 0]]))
    def invert(U):
        return sigma_z_ph @ U.T.conj() @ sigma_z_ph
    eigvs_k_inv = invert(eigvs_k)
    eigvs_minus_k = sigma_x_ph @ eigvs_k.conj() @ sigma_x_ph
    # eigvs_minus_k_inv = invert(eigvs_minus_k)

    return np.einsum(
        "mn,im,jn->ij",
        compute_eigenspace_correlator_LSWT(num_bands),
        eigvs_k,
        eigvs_minus_k,
    )


def compute_eigenspace_correlator_LSWT(num_bands):
    correlator_eigenspace = np.zeros((2*num_bands,)*2, dtype=np.complex128)
    correlator_eigenspace[0::2, 1::2] = np.eye(num_bands)

    return correlator_eigenspace


@RestoreMomenta(
    momentum_arrays_arg_idx=0,
    output_first_momentum_idx=0,
    output_is_tensor=False,
)
@CollapseMomenta(
    targets=(
        Target(arg_idx=0, first_momentum_idx=0, is_tensor=False), # eigvs
    )
)
def compute_momentum_space_correlators_LSWT(eigv_arrays):
    correlators = []

    for eigv_array in eigv_arrays:
        # (momentum idx, sublattice idx, band idx)
        assert len(eigv_array.shape) == 3

        correlators.append(np.array([
            compute_momentum_space_correlator_LSWT(eigvs_for_k)
            for idx, eigvs_for_k in enumerate(eigv_array)
        ]))

    return np.array(correlators)

