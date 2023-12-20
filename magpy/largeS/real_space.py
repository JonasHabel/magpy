import numpy as np
from magpy.util import LARGE_S_EXPANSION_COEFF
from magpy.models import Model
from magpy.interactions import Interaction



def __hc(multi_idx):
    return tuple(1-idx for idx in reversed(multi_idx))


"""
Returns all terms within the large-S expansion of order O(S^(2-{order}/2)).
These terms all contain {order} magnon operators.
"""
def compute_magnon_Hamiltonian(model: Model, order: int):
    if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for one- or two-spin interactions.")
    
    ANNIHILATOR, CREATOR = 0, 1
    C = LARGE_S_EXPANSION_COEFF # rename for brevity
    S = model.get_onsite_spin_quantum_numbers()
    rotated_spin_interactions = model.compute_rotated_interactions()

    Hamiltonians = []
    for inter in rotated_spin_interactions:
        spin_int_tensor = inter.interaction_tensor
        if len(inter.sites) == 1:
            site = inter.sites[0]
            S_i = S[0]
            magnon_BdG_tensor = np.zeros((2,)*order, dtype=np.complex128)
            
            if order == 0:
                magnon_BdG_tensor = S_i * np.array(spin_int_tensor[2])
            elif order == 2:
                magnon_BdG_tensor[CREATOR, ANNIHILATOR] = -spin_int_tensor[2]
            elif order % 2 == 1:
                t = (order-1) // 2
                idx    = (*((CREATOR, ANNIHILATOR)*t), ANNIHILATOR)
                idx_hc = __hc(idx)
                magnon_BdG_tensor[idx]    = C[t] * spin_int_tensor[0]
                magnon_BdG_tensor[idx_hc] = C[t] * spin_int_tensor[1]
            else:
                pass # no even-order terms if order >= 4
            magnon_H_i = Interaction([site]*2, magnon_BdG_tensor)
            Hamiltonians += [magnon_H_i]

        elif len(inter.sites) == 2:
            site_i, site_j = inter.sites
            S_i , S_j = S[site_i.subl_idx], S[site_j.subl_idx]
            
            if order == 1:
                magnon_BdG_tensor_i = np.sqrt(S_i) * S_j * C[0] * spin_int_tensor[0:2, 2]
                magnon_BdG_tensor_j = S_i * np.sqrt(S_j) * C[0] * spin_int_tensor[2, 0:2]
                Hamiltonians += [
                    Interaction([site_i], magnon_BdG_tensor_i),
                    Interaction([site_j], magnon_BdG_tensor_j),
                ]
            elif order >= 2 and order % 2 == 0:
                t = order // 2
                for l in range(t):
                    magnon_BdG_tensor_pp = np.zeros((2,)*order, dtype=np.complex128)
                    magnon_BdG_tensor_pm = np.zeros((2,)*order, dtype=np.complex128)
                    magnon_BdG_tensor_mp = np.zeros((2,)*order, dtype=np.complex128)
                    magnon_BdG_tensor_mm = np.zeros((2,)*order, dtype=np.complex128)
                    idx_pp = (
                        *((CREATOR, ANNIHILATOR)*(t-l-1)), ANNIHILATOR, # site i
                        *((CREATOR, ANNIHILATOR)*l), ANNIHILATOR, #site j
                    )
                    idx_pm = (
                        *((CREATOR, ANNIHILATOR)*(t-l-1)), ANNIHILATOR, # site i
                        CREATOR, *((CREATOR, ANNIHILATOR)*l), #site j
                    )
                    idx_mp = __hc(idx_pm)
                    idx_mm = __hc(idx_mm)

                    magnon_BdG_tensor_pp[idx_pp] = spin_int_tensor[0, 0]
                    magnon_BdG_tensor_pm[idx_pm] = spin_int_tensor[0, 1]
                    magnon_BdG_tensor_mp[idx_mp] = spin_int_tensor[1, 0]
                    magnon_BdG_tensor_mm[idx_mm] = spin_int_tensor[1, 1]
                    prefactor = S_i**(3/2-t+l) * S_j**(1/2-l) * C[l] * C[t-l-1]
                    magnon_BdG_tensor_pp *= prefactor
                    magnon_BdG_tensor_pm *= prefactor
                    magnon_BdG_tensor_mp *= prefactor
                    magnon_BdG_tensor_mm *= prefactor

                    sites = [site_i]*(2*(t-l)-1) + [site_j]*(2*l+1)
                    sites_hc = [site_j]*(2*l+1) + [site_i]*(2*(t-l)-1)
                    Hamiltonians += [
                        Interaction(sites, magnon_BdG_tensor_pp),
                        Interaction(sites, magnon_BdG_tensor_pm),
                        Interaction(sites_hc, magnon_BdG_tensor_mp),
                        Interaction(sites_hc, magnon_BdG_tensor_mm),
                    ]

            elif order >= 3 and order % 2 == 1:
                t = (order-1) // 2
                magnon_BdG_tensor_all_i = np.zeros((2,)*order, dtype=np.complex128)
                magnon_BdG_tensor_all_j = np.zeros((2,)*order, dtype=np.complex128)
                magnon_BdG_tensor_jj_is = np.zeros((2,)*order, dtype=np.complex128)
                magnon_BdG_tensor_ii_js = np.zeros((2,)*order, dtype=np.complex128)
                magnon_BdG_tensor_is_jj = np.zeros((2,)*order, dtype=np.complex128)
                magnon_BdG_tensor_js_ii = np.zeros((2,)*order, dtype=np.complex128)
                idx = (
                    *((CREATOR, ANNIHILATOR)*t), ANNIHILATOR, # site i or site j
                )
                idx_hc = __hc(idx)

                magnon_BdG_tensor_all_i[idx] = C[t] * S_i**(1/2-t) * S_j * spin_int_tensor[0, 2]
                magnon_BdG_tensor_all_j[idx] = C[t] * S_i * S_j**(1/2-t) * spin_int_tensor[2, 0]
                magnon_BdG_tensor_jj_is[idx] = -C[t-1] * S_i**(3/2-t) * spin_int_tensor[0, 2]
                magnon_BdG_tensor_ii_js[idx] = -C[t-1] * S_j**(3/2-t) * spin_int_tensor[2, 0]
                magnon_BdG_tensor_all_i[idx_hc] = C[t] * S_i**(1/2-t) * S_j * spin_int_tensor[1, 2]
                magnon_BdG_tensor_all_j[idx_hc] = C[t] * S_i * S_j**(1/2-t) * spin_int_tensor[2, 1]
                magnon_BdG_tensor_is_jj[idx_hc] = -C[t-1] * S_i**(3/2-t) * spin_int_tensor[1, 2]
                magnon_BdG_tensor_js_ii[idx_hc] = -C[t-1] * S_j**(3/2-t) * spin_int_tensor[2, 1]

                Hamiltonians += [
                    Interaction([site_i]*order, magnon_BdG_tensor_all_i),
                    Interaction([site_j]*order, magnon_BdG_tensor_all_j),
                    Interaction([site_j]*2 + [site_i]*(order-2), magnon_BdG_tensor_jj_is),
                    Interaction([site_i]*2 + [site_j]*(order-2), magnon_BdG_tensor_ii_js),
                    Interaction([site_i]*(order-2) + [site_j]*2, magnon_BdG_tensor_is_jj),
                    Interaction([site_j]*(order-2) + [site_i]*2, magnon_BdG_tensor_js_ii),
                ]
                

            if order == 0:
                magnon_BdG_tensor = S_i * S_j * np.array(spin_int_tensor[2, 2])
                Hamiltonians += [
                    Interaction([], magnon_BdG_tensor),
                ] 
            elif order == 2:
                magnon_BdG_tensor_ii = np.zeros((2,)*order, dtype=np.complex128)
                magnon_BdG_tensor_jj = np.zeros((2,)*order, dtype=np.complex128)
                magnon_BdG_tensor_ii[CREATOR, ANNIHILATOR] = \
                    -S_j * spin_int_tensor[2, 2]
                magnon_BdG_tensor_jj[CREATOR, ANNIHILATOR] = \
                    -S_i * spin_int_tensor[2, 2]
                Hamiltonians += [
                    Interaction([site_i]*2, magnon_BdG_tensor_ii),
                    Interaction([site_j]*2, magnon_BdG_tensor_jj),
                ] 
            elif order == 4:
                magnon_BdG_tensor_iijj = np.zeros((2,)*order, dtype=np.complex128)
                magnon_BdG_tensor_iijj[CREATOR, ANNIHILATOR, CREATOR, ANNIHILATOR] = \
                    spin_int_tensor[2, 2]
                Hamiltonians += [
                    Interaction([site_i]*2 + [site_j]*2, magnon_BdG_tensor_iijj),
                ] 
    
    return Hamiltonians


"""
model: Model
order: integer
    order of the vertex factors in 1/sqrt(S)
    e.g. order=2 -> LSWT, O(S); order=3 -> O(sqrt(S)); ...

We use the Holstein-Primakoff mapping. Each spin S' is mapped as
    S_i^z = S - a_i^dagger a_i
    S_i^+ = sqrt(2S - a_i^dagger a_i) a_i
    S_i^- = a_i^dagger sqrt(2S - a_i^dagger)
The square root terms are then expanded in powers of 1/sqrt(S):
    sqrt(2S - a_i^dagger a_i) = sqrt(2S) - a_i^dagger a_i/sqrt(8S) + ...
and terms of order <code>order</code> are then collected.

returns: list of Interaction
    each interaction is specified by a BdG-type interaction_tensor
    e.g. order=2 -> [a_i, a_i^dagger] H_bdg [a_j a_j^dagger]
"""
def compute_LSWT_Hamiltonian_real_space(model: Model):
    if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
        raise NotImplementedError("so far, only implemented for one- or two-spin interactions.")
    
    C = LARGE_S_EXPANSION_COEFF # rename for brevity
    S = model.get_onsite_spin_quantum_numbers()
    # ca = creator annihilator = a_i^† a_i
    ca = np.array([[0, 0], [1, 0]])

    rotated_spin_interactions = model.compute_rotated_interactions()
    LSWT_Hamiltonians = []
    for inter in rotated_spin_interactions:
        spin_int_tensor = inter.interaction_tensor
        if len(inter.sites) == 1:
            site = inter.sites[0]
            magnon_BdG_tensor_ii = -spin_int_tensor[2] * ca
            magnon_H_ii = Interaction([site]*2, magnon_BdG_tensor_ii)
            LSWT_Hamiltonians += [magnon_H_ii]
        elif len(inter.sites) == 2:
            site_i, site_j = inter.sites
            S_ij = np.sqrt(S[site_i.subl_idx] * S[site_j.subl_idx])
            S_ii = S[site_j.subl_idx]
            S_jj = S[site_i.subl_idx]
            magnon_BdG_tensor_ij =  S_ij * C[0]**2 * spin_int_tensor[0:2, 0:2]
            magnon_BdG_tensor_ii = -S_ii * spin_int_tensor[2, 2] * ca
            magnon_BdG_tensor_jj = -S_jj * spin_int_tensor[2, 2] * ca
            magnon_H_ij = Interaction(inter.sites, magnon_BdG_tensor_ij)
            magnon_H_ii = Interaction([site_i]*2, magnon_BdG_tensor_ii)
            magnon_H_jj = Interaction([site_j]*2, magnon_BdG_tensor_jj)
            LSWT_Hamiltonians += [
                magnon_H_ij, magnon_H_ii, magnon_H_jj,
            ]   
    
    return LSWT_Hamiltonians



def compute_interaction_Hamiltonian(model: Model, order):
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