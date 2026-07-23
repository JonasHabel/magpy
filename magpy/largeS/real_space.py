import numpy as np
from magpy.util import LARGE_S_EXPANSION_COEFF as C # rename for brevity
from magpy.util import ENERGY_EPS
from magpy.models import Model
from magpy.interactions import Interaction, compress, compute_rotated_interactions
from itertools import combinations


ANNIHILATOR, CREATOR = 0, 1

def __hc(multi_idx):
    return tuple(1-idx for idx in reversed(multi_idx))

"""
Returns all terms within the large-S expansion of order O(S^(2-{order}/2)).
These terms all contain {order} magnon operators.
"""
def compute_magnon_Hamiltonian(model: Model, order: int, output_compression=None):
    # if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
    #     raise NotImplementedError("so far, only implemented for one- or two-spin interactions.")
    
    S = model.get_onsite_spin_quantum_numbers()
    inter_by_sites = compress(model.interactions)
    rotated_spin_interactions = compute_rotated_interactions(
        inter_by_sites, model.compute_ground_state_rotation_matrices()
    )

    Hamiltonians = []
    for inter in rotated_spin_interactions:
        spin_int_tensor = inter.interaction_tensor
        if len(inter.sites) == 1:
            Hamiltonians += compute_magnon_Hamiltonian_1site_interaction(inter, S, order)
        elif len(inter.sites) == 2:
            Hamiltonians += compute_magnon_Hamiltonian_2site_interaction(inter, S, order)
        elif len(inter.sites) == 4:
            if order == 0:
                Hamiltonians += compute_magnon_Hamiltonian_0th_order(inter, S)
            elif order == 1:
                Hamiltonians += compute_magnon_Hamiltonian_1st_order(inter, S)
            elif order == 2:
                Hamiltonians += compute_magnon_Hamiltonian_2nd_order(inter, S)
            elif order == 3:
                Hamiltonians += compute_magnon_Hamiltonian_3rd_order(inter, S)
            elif order == 4:
                Hamiltonians += compute_magnon_Hamiltonian_4th_order(inter, S)
            else:
                raise ValueError(f"Unsupported order ({order}) for {len(inter.sites)}-site interaction.")
        else:
            raise ValueError(f"Unsupported number of sites ({len(inter.sites)}) for interaction.")
    
    Hamiltonians = list(filter(
        lambda H: np.any(np.abs(H.interaction_tensor) > ENERGY_EPS), # np.zeros(H.interaction_tensor.shape)),
        Hamiltonians))
    
    if output_compression is not None:
        Hamiltonians = compress(Hamiltonians, **output_compression)

    return Hamiltonians



# up to arbitrary order in 1/S
def compute_magnon_Hamiltonian_1site_interaction(inter: Interaction, S, order: int):
    spin_int_tensor = inter.interaction_tensor
    site = inter.sites[0]
    S_i = S[0]
    magnon_BdG_tensor = np.zeros((2,)*order, dtype=np.complex128)
    
    if order == 0:
        magnon_BdG_tensor = S_i * np.array(spin_int_tensor[2])
    elif order == 2:
        magnon_BdG_tensor[CREATOR, ANNIHILATOR] = -spin_int_tensor[2]
    elif order % 2 == 1:
        t = (order-1) // 2
        idx1    = (*((CREATOR, ANNIHILATOR)*t), ANNIHILATOR)
        idx1_hc = __hc(idx1)
        magnon_BdG_tensor[idx1]    = S_i**(1/2 - t) * C[t] * spin_int_tensor[0]
        magnon_BdG_tensor[idx1_hc] = S_i**(1/2 - t) * C[t] * spin_int_tensor[1]
    else:
        return [] # no even-order terms if order >= 4

    magnon_H_i = Interaction([site]*order, magnon_BdG_tensor)
    return [magnon_H_i]



# up to arbitrary order in 1/S
def compute_magnon_Hamiltonian_2site_interaction(inter: Interaction, S, order: int):
    spin_int_tensor = inter.interaction_tensor
    site_i, site_j = inter.sites
    S_i , S_j = S[site_i.subl_idx], S[site_j.subl_idx]
    Hamiltonians = []

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
            idx_mm = __hc(idx_pp)

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
        idx1 = (
            *((CREATOR, ANNIHILATOR)*t), ANNIHILATOR,
        )
        idx2 = (
            *((CREATOR, ANNIHILATOR)*(t-1)), ANNIHILATOR, CREATOR, ANNIHILATOR,
        )
        idx1_hc = __hc(idx1)
        idx2_hc = __hc(idx2)

        magnon_BdG_tensor_all_i[idx1] = C[t] * S_i**(1/2-t) * S_j * spin_int_tensor[0, 2]
        magnon_BdG_tensor_all_j[idx1] = C[t] * S_i * S_j**(1/2-t) * spin_int_tensor[2, 0]
        magnon_BdG_tensor_jj_is[idx2] = -C[t-1] * S_i**(3/2-t) * spin_int_tensor[0, 2]
        magnon_BdG_tensor_ii_js[idx1] = -C[t-1] * S_j**(3/2-t) * spin_int_tensor[2, 0]
        magnon_BdG_tensor_all_i[idx1_hc] = C[t] * S_i**(1/2-t) * S_j * spin_int_tensor[1, 2]
        magnon_BdG_tensor_all_j[idx1_hc] = C[t] * S_i * S_j**(1/2-t) * spin_int_tensor[2, 1]
        magnon_BdG_tensor_is_jj[idx2_hc] = -C[t-1] * S_i**(3/2-t) * spin_int_tensor[1, 2]
        magnon_BdG_tensor_js_ii[idx1_hc] = -C[t-1] * S_j**(3/2-t) * spin_int_tensor[2, 1]

        Hamiltonians += [
            Interaction([site_i]*order, magnon_BdG_tensor_all_i),
            Interaction([site_j]*order, magnon_BdG_tensor_all_j),
            Interaction([site_i]*(order-2) + [site_j]*2, magnon_BdG_tensor_jj_is),
            Interaction([site_i]*2 + [site_j]*(order-2), magnon_BdG_tensor_ii_js),
            Interaction([site_j]*2 + [site_i]*(order-2), magnon_BdG_tensor_is_jj),
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



# for interactions acting on arbitrarily many sites
def compute_magnon_Hamiltonian_0th_order(inter: Interaction, S):
    spin_int_tensor = inter.interaction_tensor
    Ss = [S[site.subl_idx] for site in inter.sites]
    prod_S = np.prod(Ss)

    tensor_idx = tuple(2 for n in range(len(inter.sites)))
    magnon_BdG_tensor = prod_S * np.array(spin_int_tensor[tensor_idx])

    return [Interaction([], magnon_BdG_tensor)] 



def compute_magnon_Hamiltonian_1st_order(inter: Interaction, S):
    spin_int_tensor = inter.interaction_tensor
    Ss = [S[site.subl_idx] for site in inter.sites]
    prod_S = np.prod(Ss)
    Hamiltonians = []

    for nsite, (site, S) in enumerate(zip(inter.sites, Ss)):
        tensor_idx = tuple((slice(0, 2) if n == nsite else 2) for n in range(len(inter.sites)))
        magnon_BdG_tensor = prod_S / np.sqrt(S) * C[0] * spin_int_tensor[tensor_idx]
        Hamiltonians += [
            Interaction([site], magnon_BdG_tensor),
        ]
    
    return Hamiltonians



def compute_magnon_Hamiltonian_2nd_order(inter: Interaction, S):
    spin_int_tensor = inter.interaction_tensor
    Ss = [S[site.subl_idx] for site in inter.sites]
    prod_S = np.prod(Ss)
    Hamiltonians = []

    # S^\pm S^\pm terms
    site_combis = combinations(inter.sites, 2)
    S_combis = combinations(Ss, 2)
    idx_combis = combinations(range(len(inter.sites)), 2)
    for (idx_combi, site_combi, S_combi) in zip(idx_combis, site_combis, S_combis):
        tensor_idx = tuple((slice(0, 2) if n in idx_combi else 2) for n in range(len(inter.sites)))
        magnon_BdG_tensor = prod_S / np.sqrt(np.prod(S_combi)) * C[0]**2 * spin_int_tensor[tensor_idx]
        Hamiltonians += [
            Interaction(list(site_combi), magnon_BdG_tensor),
        ]

    # S^z S^z terms (= S a^† a)
    for nsite, (site, Ssite) in enumerate(zip(inter.sites, Ss)):
        tensor_idx = tuple(2 for n in range(len(inter.sites)))
        magnon_BdG_tensor = np.zeros((2, 2))
        magnon_BdG_tensor[CREATOR, ANNIHILATOR] = -prod_S / Ssite * spin_int_tensor[tensor_idx]
        Hamiltonians += [
            Interaction([site]*2, magnon_BdG_tensor),
        ]

    return Hamiltonians



def compute_magnon_Hamiltonian_3rd_order(inter: Interaction, S):
    spin_int_tensor = inter.interaction_tensor
    Ss = [S[site.subl_idx] for site in inter.sites]
    prod_S = np.prod(Ss)
    Hamiltonians = []

    # S^\pm S^\pm S^\pm terms
    site_combis = combinations(inter.sites, 3)
    S_combis = combinations(Ss, 3)
    idx_combis = combinations(range(len(inter.sites)), 3)
    for (idx_combi, site_combi, S_combi) in zip(idx_combis, site_combis, S_combis):
        tensor_idx = tuple((slice(0, 2) if n in idx_combi else 2) for n in range(len(inter.sites)))
        magnon_BdG_tensor = prod_S / np.sqrt(np.prod(S_combi)) * C[0]**3 * spin_int_tensor[tensor_idx]
        Hamiltonians += [
            Interaction(list(site_combi), magnon_BdG_tensor),
        ]

    # S^\pm S^z S^z terms (= S^\pm (a^† a) S)
    site_combis = combinations(inter.sites, 2)
    S_combis = combinations(Ss, 2)
    idx_combis = combinations(range(len(inter.sites)), 2)
    for (idx_combi, site_combi, S_combi) in zip(idx_combis, site_combis, S_combis):
        tensor_idx1 = tuple((slice(0, 2) if n == idx_combi[0] else 2) for n in range(len(inter.sites)))
        tensor_idx2 = tuple((slice(0, 2) if n == idx_combi[1] else 2) for n in range(len(inter.sites)))
        magnon_BdG_tensor1 = np.zeros((2, 2, 2))
        magnon_BdG_tensor2 = np.zeros((2, 2, 2))
        magnon_BdG_tensor1[:, CREATOR, ANNIHILATOR] = -prod_S / S_combi[1] / np.sqrt(S_combi[0]) * C[0] * spin_int_tensor[tensor_idx1]
        magnon_BdG_tensor2[CREATOR, ANNIHILATOR, :] = -prod_S / S_combi[0] / np.sqrt(S_combi[1]) * C[0] * spin_int_tensor[tensor_idx2]
        Hamiltonians += [
            Interaction([site_combi[0], site_combi[1], site_combi[1]], magnon_BdG_tensor1),
            Interaction([site_combi[0], site_combi[0], site_combi[1]], magnon_BdG_tensor2),
        ]

    # S^\pm S^z S^z terms (= S^\pm S S with higher-order expansion of the square root)
    for nsite, (site, S) in enumerate(zip(inter.sites, S)):
        tensor_idx_annihil = tuple((ANNIHILATOR if n == nsite else 2) for n in range(len(inter.sites)))
        tensor_idx_creator = tuple((CREATOR if n == nsite else 2) for n in range(len(inter.sites)))
        magnon_BdG_tensor = np.zeros((2, 2, 2))
        magnon_BdG_tensor[CREATOR, CREATOR, ANNIHILATOR] = prod_S / np.sqrt(S**3) * C[1] * spin_int_tensor[tensor_idx_creator]
        magnon_BdG_tensor[CREATOR, ANNIHILATOR, ANNIHILATOR] = prod_S / np.sqrt(S**3) * C[1] * spin_int_tensor[tensor_idx_annihil]
        Hamiltonians += [
            Interaction([site]*3, magnon_BdG_tensor),
        ]

    return Hamiltonians



def compute_magnon_Hamiltonian_4th_order(inter: Interaction, S):
    spin_int_tensor = inter.interaction_tensor
    Ss = [S[site.subl_idx] for site in inter.sites]
    prod_S = np.prod(Ss)
    Hamiltonians = []

    # S^\pm S^\pm S^\pm S^\pm terms
    site_combis = combinations(inter.sites, 4)
    S_combis = combinations(Ss, 4)
    idx_combis = combinations(range(len(inter.sites)), 4)
    for (idx_combi, site_combi, S_combi) in zip(idx_combis, site_combis, S_combis):
        tensor_idx = tuple((slice(0, 2) if n in idx_combi else 2) for n in range(len(inter.sites)))
        magnon_BdG_tensor = prod_S / np.sqrt(np.prod(S_combi)) * C[0]**4 * spin_int_tensor[tensor_idx]
        Hamiltonians += [
            Interaction(list(site_combi), magnon_BdG_tensor),
        ]

    # S^\pm S^\pm S^z S^z terms (= S^\pm S^\pm (a^† a) S)
    site_combis = combinations(inter.sites, 3)
    S_combis = combinations(Ss, 3)
    idx_combis = combinations(range(len(inter.sites)), 3)
    for (idx_combi, site_combi, S_combi) in zip(idx_combis, site_combis, S_combis):
        for m in range(3):  # m is the position of (a^† a) relative to S^\pm S^\pm (m = 0 for (a^† a) before S^\pm S^\pm, m = 1 for (a^† a) between S^\pm and S^\pm, m = 2 for (a^† a) after S^\pm S^\pm)
            idx_combi_Spm = [idx_combi[l] for l in range(3) if l != m]
            S_combi_Spm = [S_combi[l] for l in range(3) if l != m]
            tensor_idx = tuple((slice(0, 2) if n in idx_combi_Spm else 2) for n in range(len(inter.sites)))
            magnon_BdG_tensor = np.zeros((2, 2, 2, 2))
            BdG_idx = tuple([slice(None) for l in range(m)] + [CREATOR, ANNIHILATOR] + [slice(None) for l in range(m+1, 3)])
            magnon_BdG_tensor[BdG_idx] = -prod_S / np.prod(np.sqrt(S_combi_Spm) * S_combi[m]) * C[0]**2 * spin_int_tensor[tensor_idx]
            Hamiltonians += [
                Interaction(list(site_combi), magnon_BdG_tensor),
            ]

    # S^\pm S^\pm S^z S^z terms (= S^\pm S^\pm S S with higher-order expansion of the square root)
    site_combis = combinations(inter.sites, 2)
    S_combis = combinations(Ss, 2)
    idx_combis = combinations(range(len(inter.sites)), 2)
    for (idx_combi, site_combi, S_combi) in zip(idx_combis, site_combis, S_combis):
        tensor_idx1_creator = tuple((CREATOR if n == idx_combi[0] else slice(0, 2) if n == idx_combi[1] else 2) for n in range(len(inter.sites)))
        tensor_idx1_annihil = tuple((ANNIHILATOR if n == idx_combi[0] else slice(0, 2) if n == idx_combi[1] else 2) for n in range(len(inter.sites)))
        tensor_idx2_creator = tuple((CREATOR if n == idx_combi[1] else slice(0, 2) if n == idx_combi[0] else 2) for n in range(len(inter.sites)))
        tensor_idx2_annihil = tuple((ANNIHILATOR if n == idx_combi[1] else slice(0, 2) if n == idx_combi[0] else 2) for n in range(len(inter.sites)))
        magnon_BdG_tensor1 = np.zeros((2, 2, 2, 2))
        magnon_BdG_tensor2 = np.zeros((2, 2, 2, 2))
        magnon_BdG_tensor1[CREATOR, CREATOR, ANNIHILATOR, :] = prod_S / np.sqrt(S_combi[0]**3 * S_combi[1]) * C[0] * C[1] * spin_int_tensor[tensor_idx1_creator]
        magnon_BdG_tensor1[CREATOR, ANNIHILATOR, ANNIHILATOR, :] = prod_S / np.sqrt(S_combi[0]**3 * S_combi[1]) * C[0] * C[1] * spin_int_tensor[tensor_idx1_annihil]
        magnon_BdG_tensor2[:, CREATOR, CREATOR, ANNIHILATOR] = prod_S / np.sqrt(S_combi[0] * S_combi[1]**3) * C[0] * C[1] * spin_int_tensor[tensor_idx2_creator]
        magnon_BdG_tensor2[:, CREATOR, ANNIHILATOR, ANNIHILATOR] = prod_S / np.sqrt(S_combi[0] * S_combi[1]**3) * C[0] * C[1] * spin_int_tensor[tensor_idx2_annihil]
        Hamiltonians += [
            Interaction([site_combi[0], site_combi[0], site_combi[0], site_combi[1]], magnon_BdG_tensor1),
            Interaction([site_combi[0], site_combi[1], site_combi[1], site_combi[1]], magnon_BdG_tensor2),
        ]

    # S^z S^z S^z S^z terms (= (a^† a) (a^† a) S S)
    site_combis = combinations(inter.sites, 2)
    S_combis = combinations(Ss, 2)
    idx_combis = combinations(range(len(inter.sites)), 2)
    for (idx_combi, site_combi, S_combi) in zip(idx_combis, site_combis, S_combis):
        tensor_idx = tuple(2 for n in range(len(inter.sites)))
        magnon_BdG_tensor = np.zeros((2, 2, 2, 2))
        magnon_BdG_tensor[CREATOR, ANNIHILATOR, CREATOR, ANNIHILATOR] = prod_S / np.prod(S_combi) * spin_int_tensor[tensor_idx]
        Hamiltonians += [
            Interaction([site_combi[0], site_combi[0], site_combi[1], site_combi[1]], magnon_BdG_tensor),
        ]

    return Hamiltonians