import numpy as np
from operator import itemgetter
from magpy.models import Model
from magpy.lattice import ReciprocalLattice
from magpy.interactions import Interaction
from magpy.util import BOGO_METRIC, LARGE_S_EXPANSION_COEFF
from magpy.largeS import momentum_space
from magpy.largeS.util import get_real_space_magnon_Hamiltonian



def compute_LSWT_Hamiltonian_momentum_space_BdG(
        model: Model, k, LSWT_Hamiltonian_real_space=None):
    if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for LSWT of one- or two-spin interactions.")
    
    if model.lattice.dim >= 1 and \
        model.lattice.dim != len(k):
        raise Exception(f"dimension of momentum vector " \
                        + f"{len(k)} must equal the "\
                        + f"lattice dimension {model.lattice.dim}")
    
    LSWT_Hamiltonian_real_space = get_real_space_magnon_Hamiltonian(
        LSWT_Hamiltonian_real_space, model, order=2)

    num_sites_unit_cell = model.lattice.num_sites_unit_cell
    magnon_H_k = \
        momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(
            model, ks=np.array([k]), 
            interaction_Hamiltonian_real_space=LSWT_Hamiltonian_real_space)
    
    magnon_H_minusk = \
        momentum_space.compute_magnon_Hamiltonian_with_momentum_conservation(
            model, ks=np.array([-k]), 
            interaction_Hamiltonian_real_space=LSWT_Hamiltonian_real_space)
    
    sigma_x = np.array([[0, 1], [1, 0]])
    # of the form [a_k^†, a_{-k}] H_BdG [a_k, a_{-k}^†]
    magnon_H_BdG = 0.5 * np.kron(np.eye(num_sites_unit_cell), sigma_x) \
        @ (magnon_H_k + magnon_H_minusk.T)

    # make BdG Hamiltonians hermitian
    # magnon_H_BdG += np.conj(magnon_H_BdG.T)
    # magnon_H_BdG /= 2

    return magnon_H_BdG