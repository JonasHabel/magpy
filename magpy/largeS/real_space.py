import numpy as np
from magpy.util import LARGE_S_EXPANSION_COEFF, ENERGY_EPS
from magpy.models import Model
from magpy.interactions import Interaction, compress, compute_rotated_interactions
from itertools import combinations


def __hc(multi_idx):
    return tuple(1-idx for idx in reversed(multi_idx))


"""
Returns all terms within the large-S expansion of order O(S^(2-{order}/2)).
These terms all contain {order} magnon operators.
"""
def compute_magnon_Hamiltonian(model: Model, order: int, output_compression=None):
    # if any(map(lambda inter: len(inter.sites) not in [1, 2], model.interactions)):
    #     raise NotImplementedError("so far, only implemented for one- or two-spin interactions.")
    
    ANNIHILATOR, CREATOR = 0, 1
    C = LARGE_S_EXPANSION_COEFF # rename for brevity
    S = model.get_onsite_spin_quantum_numbers()
    inter_by_sites = compress(model.interactions)
    rotated_spin_interactions = compute_rotated_interactions(
        inter_by_sites, model.compute_ground_state_rotation_matrices()
    )

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
                idx1    = (*((CREATOR, ANNIHILATOR)*t), ANNIHILATOR)
                idx1_hc = __hc(idx1)
                magnon_BdG_tensor[idx1]    = C[t] * spin_int_tensor[0]
                magnon_BdG_tensor[idx1_hc] = C[t] * spin_int_tensor[1]
            else:
                pass # no even-order terms if order >= 4
            magnon_H_i = Interaction([site]*order, magnon_BdG_tensor)
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

        elif len(inter.sites) == 4:
            site_i, site_j, site_k, site_l = inter.sites
            Ss = [S[site.subl_idx] for site in inter.sites]
            prod_S = np.prod(Ss)

            if order == 0:
                magnon_BdG_tensor = prod_S * np.array(spin_int_tensor[2, 2, 2, 2])
                Hamiltonians += [
                    Interaction([], magnon_BdG_tensor),
                ] 
            elif order == 1:
                for nsite, (site, S) in enumerate(zip(inter.sites, S)):
                    tensor_idx = tuple((slice(0, 2) if n == nsite else 2) for n in range(len(inter.sites)))
                    magnon_BdG_tensor = prod_S / np.sqrt(S) * C[0] * spin_int_tensor[tensor_idx]
                    Hamiltonians += [
                        Interaction([site], magnon_BdG_tensor),
                    ]
            elif order == 2:
                site_combis = combinations(inter.sites, 2)
                S_combis = combinations(Ss, 2)
                idx_combis = combinations(range(len(inter.sites)), 2)
                for (idx_combi, site_combi, S_combi) in zip(idx_combis, site_combis, S_combis):
                    tensor_idx = tuple((slice(0, 2) if n in idx_combi else 2) for n in range(len(inter.sites)))
                    magnon_BdG_tensor = prod_S / np.sqrt(np.prod(S_combi)) * C[0]**2 * spin_int_tensor[tensor_idx]
                    Hamiltonians += [
                        Interaction(site_combi, magnon_BdG_tensor),
                    ]

                for nsite, (site, Ssite) in enumerate(zip(inter.sites, Ss)):
                    magnon_BdG_tensor = np.zeros((2, 2))
                    magnon_BdG_tensor[CREATOR, ANNIHILATOR] = -prod_S / Ssite * spin_int_tensor[2, 2, 2, 2]
                    Hamiltonians += [
                        Interaction([site]*2, magnon_BdG_tensor),
                    ]
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
