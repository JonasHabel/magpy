import numpy as np
from ..util import LARGE_S_EXPANSION_COEFF
from ..models import Model
from ..interactions import Interaction



"""
returns: list of numpy array
    a list of tensors of coeff.s of the {order}-th order interaction vertices
    in real space
    e.g. order=3, returns the coefficients of
        a^† a^† a
        a^† a a
    (there are no terms like a^† a^† a^† or a a a before plugging in the
    Bogoliubov trafo)
"""
def compute_interaction_Hamiltonian_real_space(model: Model, order):
    if order not in [3] or any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for cubic vertices of one- or two-spin interactions.")
    
    C = LARGE_S_EXPANSION_COEFF # rename for brevity
    S = model.get_onsite_spin_quantum_numbers()
    # caa = creator annihilator annihilator = a^† a a
    caa = np.zeros((2, 2, 2))
    caa[1, 0, 0] = 1
    # cca = creator creator annihilator = a^† a^† a
    cca = np.zeros((2, 2, 2))
    cca[1, 1, 0] = 1

    rotated_spin_interactions = model.compute_rotated_interactions()
    magnon_Hamiltonians = []
    for inter in rotated_spin_interactions:
        spin_int_tensor = inter.interaction_tensor
        if len(inter.sites) == 1:
            site = inter.sites[0]
            magnon_BdG_tensor_iii = C[1] * (spin_int_tensor[0] * caa \
                                          + spin_int_tensor[1] * cca)
            magnon_H_iii = Interaction([site]*3, magnon_BdG_tensor_iii)
            magnon_Hamiltonians += [magnon_H_iii]
        elif len(inter.sites) == 2:
            site_i, site_j = inter.sites
            S_iii = S[site_j.subl_idx] / np.sqrt(S[site_i.subl_idx])
            S_jjj = S[site_i.subl_idx] / np.sqrt(S[site_j.subl_idx])
            S_jij = S_ijj = np.sqrt(S[site_i.subl_idx])
            S_iij = S_iji = np.sqrt(S[site_j.subl_idx])
            magnon_BdG_tensor_iii = S_iii * C[1] * (spin_int_tensor[0, 2] * caa\
                                                  + spin_int_tensor[1, 2] * cca)
            magnon_BdG_tensor_jjj = S_jjj * C[1] * (spin_int_tensor[2, 0] * caa\
                                                  + spin_int_tensor[2, 1] * cca)
            magnon_BdG_tensor_jij = -S_jij * C[0] * spin_int_tensor[0, 2] * caa
            magnon_BdG_tensor_ijj = -S_ijj * C[0] * spin_int_tensor[1, 2] * cca
            magnon_BdG_tensor_iij = -S_iij * C[0] * spin_int_tensor[2, 0] * caa
            magnon_BdG_tensor_iji = -S_iji * C[0] * spin_int_tensor[2, 1] * cca
            magnon_H_iii = Interaction([inter.sites[0]]*3, magnon_BdG_tensor_iii)
            magnon_H_jjj = Interaction([inter.sites[1]]*3, magnon_BdG_tensor_jjj)
            magnon_H_jij = Interaction([inter.sites[n] for n in [1, 0, 1]], magnon_BdG_tensor_jij)
            magnon_H_ijj = Interaction([inter.sites[n] for n in [0, 1, 1]], magnon_BdG_tensor_ijj)
            magnon_H_iij = Interaction([inter.sites[n] for n in [0, 0, 1]], magnon_BdG_tensor_iij)
            magnon_H_iji = Interaction([inter.sites[n] for n in [0, 1, 0]], magnon_BdG_tensor_iji)
            magnon_Hamiltonians += [
                magnon_H_iii, magnon_H_jjj, magnon_H_jij,
                magnon_H_ijj, magnon_H_iij, magnon_H_iji,
            ]
    
    return magnon_Hamiltonians

