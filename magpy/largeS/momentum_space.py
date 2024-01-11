import numpy as np
from magpy.models import Model
from magpy.largeS import real_space


def compute_magnon_Hamiltonian(model: Model, order, ks,
        interaction_Hamiltonian_real_space=None):
    if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for one- or two-spin interactions.")
    
    if model.lattice.dim >= 1 and model.lattice.dim != ks.shape[-1]:
        raise Exception(f"dimension of each momentum vector " \
                      + f"{ks.shape[-1]} must equal the " \
                      + f"lattice dimension {model.lattice.dim}")
    
    if order != ks.shape[0]:
        raise Exception(f"number of momentum vectors " \
                      + f"{ks.shape[0]} must equal the " \
                      + f"order of the vertex")

    magnon_Hs_real_space = interaction_Hamiltonian_real_space \
        if interaction_Hamiltonian_real_space is not None \
        else real_space.compute_magnon_Hamiltonian(model, order)

    H_dim = 2*model.lattice.num_sites_unit_cell
    magnon_H_mom_space = np.zeros((H_dim,) * order, dtype=np.complex128)

    ks = ks.astype(np.float64)
    for coupling in magnon_Hs_real_space:
        subl_idxs = [site.subl_idx for site in coupling.sites]
        bravais_vecs = np.array([
            model.lattice.to_canonical_basis(site.bravais_coords) \
            for site in coupling.sites
        ]).reshape(ks.shape)    # reshape only necessary for order == 0 case
        phase = np.exp(1j * np.einsum("ij,ij", ks, bravais_vecs))
        idx = tuple(slice(2*subl_idx, 2*(subl_idx+1)) for subl_idx in subl_idxs)
        magnon_H_mom_space[idx] += phase * coupling.interaction_tensor
        
    # make BdG Hamiltonians hermitian
    # magnon_H_mom_space += \
    #     np.conj(np.transpose(magnon_H_mom_space, axes=[0, 2, 1]))
    # magnon_H_mom_space /= 2

    return magnon_H_mom_space