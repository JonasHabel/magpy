import numpy as np
from .models import *
from .lattice import BravaisLattice
from scipy.integrate import solve_ivp
from numba import njit


@njit
def __shift_bravais_coord_pbc(unit_cell_bravais_coord, site_bravais_coords,
                              sizes):
    return ((unit_cell_bravais_coord + site_bravais_coords) % sizes) \
        .astype(np.int64)

@njit
def __convert_to_flat_index(unit_cell_bravais_coord,
                            inter_site_bravais_coord, inter_site_subl_idx,
                            sizes, num_sites_unit_cell):
    inter_site_absolute_bravais_coord =\
        __shift_bravais_coord_pbc(unit_cell_bravais_coord,
                                  inter_site_bravais_coord, sizes)

    flat_idx = inter_site_subl_idx
    factor = num_sites_unit_cell
    for i in range(len(inter_site_absolute_bravais_coord)-1, -1, -1):
        flat_idx += factor * inter_site_absolute_bravais_coord[i]
        factor *= sizes[i]
    return flat_idx


@njit
def compute_effective_field_for_interaction_jit(
        inter_sites_bravais_coords, inter_sites_subl_idxs, interaction_tensor,
        unit_cell_bravais_coords, spin_config, eff_field_out, sizes,
        num_sites_unit_cell):
    num_inter_sites = len(inter_sites_subl_idxs)
    site_idxs = np.arange(num_inter_sites)

    for unit_cell_bravais_coord in unit_cell_bravais_coords:
        flat_idxs_for_inter_sites = np.array([
            __convert_to_flat_index(
                    unit_cell_bravais_coord,
                    inter_sites_bravais_coords[site_idx],
                    inter_sites_subl_idxs[site_idx],
                    sizes, num_sites_unit_cell)
            for site_idx in site_idxs
        ])

        participating_spins = np.zeros((num_inter_sites, 3), dtype=np.float64)
        for n, flat_idx in enumerate(flat_idxs_for_inter_sites):
            participating_spins[n] = spin_config[flat_idx]

        if num_inter_sites == 1:
            eff_field_at_site_1 = -interaction_tensor.reshape((3,))
            eff_field_out[flat_idxs_for_inter_sites[0]] += eff_field_at_site_1
        elif num_inter_sites == 2:
            eff_field_at_site_1 = \
                -interaction_tensor.dot(participating_spins[1])#.reshape((3,))
            eff_field_at_site_2 = \
                -interaction_tensor.T.dot(participating_spins[0])#.reshape((3,))
            
            eff_field_out[flat_idxs_for_inter_sites[0]] += eff_field_at_site_1
            eff_field_out[flat_idxs_for_inter_sites[1]] += eff_field_at_site_2
        else:
            raise Exception(">=3-spin interactions not yet supported.")



def compute_effective_field_for_interaction(
        inter_sites_bravais_coords, inter_sites_subl_idxs, interaction_tensor,
        unit_cell_bravais_coords, spin_config, eff_field_out,
        sizes, num_sites_unit_cell):
    num_inter_sites = len(inter_sites_subl_idxs)
    site_idxs = np.arange(num_inter_sites)
    
    for unit_cell_bravais_coord in unit_cell_bravais_coords:
        flat_idxs_for_inter_sites = np.array([
            __convert_to_flat_index(
                    unit_cell_bravais_coord,
                    inter_sites_bravais_coords[site_idx],
                    inter_sites_subl_idxs[site_idx],
                    sizes, num_sites_unit_cell)
            for site_idx in site_idxs
        ])
        participating_spins = np.array([
            spin_config[flat_idx] for flat_idx in flat_idxs_for_inter_sites
        ])
        
        for site_idx, site_flat_idx in zip(
                site_idxs, flat_idxs_for_inter_sites):
            other_site_idxs = site_idxs[
                np.r_[0:site_idx, site_idx+1:num_inter_sites]]
            other_participating_spins = participating_spins[
                np.r_[0:site_idx, site_idx+1:num_inter_sites]]
            
            einsum_idxs_int_tensor = util.generate_einsum_indices(site_idxs)
            einsum_str = "".join(einsum_idxs_int_tensor)
            if num_inter_sites >= 2:
                einsum_idxs_spins = util.generate_einsum_indices(other_site_idxs)
                einsum_str += "," + ",".join(einsum_idxs_spins)

            eff_field_at_site = -np.einsum(einsum_str,
                                 interaction_tensor,
                                 *other_participating_spins)
            
            eff_field_out[site_flat_idx] += eff_field_at_site

"""
compute the effective field B_eff = -dH / dS_i for each site i

dims: tuple of ints specifying the no. of unit cells in each periodic direction
spin_config: numpy array, shape = (*dims, num_sites_unit_cell, 3)

returns: numpy array
    shape = (*dims, num_sites_unit_cell, 3)
"""
def compute_effective_field(model: Model, spin_config,
                            sizes, num_sites_unit_cell, use_jit=True):
    num_sites_unit_cell = model.lattice.num_sites_unit_cell
    num_unit_cells = np.prod(sizes)
    num_sites_total = num_unit_cells * num_sites_unit_cell

    unit_cell_bravais_coords = \
        model.lattice.sample_Bravais_lattice_in_Bravais_coords(sizes)

    eff_field = np.zeros((num_sites_total, 3), dtype=float)

    for inter in model.interactions:
        inter_sites_bravais_coords = np.array([
            site.bravais_coords for site in inter.sites
        ], dtype=float)
        inter_sites_subl_idxs = np.array([
            site.subl_idx for site in inter.sites
        ], dtype=int)

        f = compute_effective_field_for_interaction_jit if use_jit else \
            compute_effective_field_for_interaction
        f(inter_sites_bravais_coords, inter_sites_subl_idxs,
         inter.interaction_tensor, unit_cell_bravais_coords,
         spin_config, eff_field, sizes, num_sites_unit_cell)

    return eff_field




def simulate_LLG(model: Model, sizes, time_span, num_times, init_spin_config,
                 damping=0.0, boundary_conditions={}, use_jit=True):
    sizes = np.array(sizes, dtype=int)
    num_sites_unit_cell = model.lattice.num_sites_unit_cell
    num_unit_cells = np.prod(sizes)
    num_sites_total = num_unit_cells * num_sites_unit_cell
    bc_dS_dt = boundary_conditions.get("dS/dt")

    def spin_config_time_derivative(time, spin_config):
        # do LLG step
        spin_config = spin_config.reshape((num_sites_total, 3))
        effective_field = compute_effective_field(
            model, spin_config, sizes, num_sites_unit_cell, use_jit)
        S_cross_B_eff = np.cross(spin_config, effective_field)
        dS_dt = S_cross_B_eff - damping*np.cross(spin_config, S_cross_B_eff)

        # apply dS/dt boundary conditions
        if bc_dS_dt is not None:
            S_reshaped = spin_config.reshape(
                (*sizes, num_sites_unit_cell, 3))
            dS_dt_reshaped = dS_dt.reshape(
                (*sizes, num_sites_unit_cell, 3))
            for coord, dS_dt_bc_func in zip(
                    bc_dS_dt["coords"], bc_dS_dt["func"]):
                flat_idx = __convert_to_flat_index(
                    np.zeros((len(sizes),), dtype=int), coord[:-1], coord[-1],
                    sizes, num_sites_unit_cell)
                dS_dt_bc = dS_dt_bc_func(
                    time, tuple(coord), S_reshaped, dS_dt_reshaped)
                if dS_dt_bc is not None:
                    dS_dt[flat_idx] = dS_dt_bc

        return dS_dt.reshape((3*num_sites_total))
    
    init_spin_config = init_spin_config.reshape((num_sites_total*3))
    solution = solve_ivp(
        spin_config_time_derivative, time_span, init_spin_config,
        t_eval=np.linspace(*time_span, num_times))
    
    solution["y"] = solution["y"] \
        .reshape((*sizes, num_sites_unit_cell, 3, num_times))
    return solution["t"], solution["y"]

    

def sample_initial_config(
        model: Model, sizes, time_span, num_times, sampler, num_samples,
        boundary_conditions={}, damping=0.0, use_jit=True):
    S = np.zeros(
        (num_samples, *sizes, model.lattice.num_sites_unit_cell, 3, num_times))
    
    for i in range(num_samples):
        init_spin_config = sampler()
        print(f"Running sample {i+1} / {num_samples}")
        times, S[i] = simulate_LLG(model, sizes, time_span, num_times,
                            init_spin_config, damping, boundary_conditions,
                            use_jit=use_jit)
        
    return times, S



"""
S: numpy array (*sizes, num_sites_unit_cell, 3, num_times)
points: list of tuples
    The number of tuples determines the order of the correlator
    (n tuples => n-point correlator).
    Each tuple is of the form (...bravais_coords, subl_coord, spin-x/y/z, time).

Example for a 1D lattice:
    points == [(0, 1, 2, 0), (slice(None), 0, 0, slice(None))]
    computes the C(X, T) = <S_{x=0,subl=B}^z(t=0) S_{x=X,subl=A}^x(t=T)>
    two-point correlator
"""
def compute_n_point_function(S, points, out=None):
    if out is not None:
        assert len(points) == len(out.shape)
    
    S_evaluated_at_points = [
        S[(*pt, )] for pt in points
    ]

    class LetterGenerator:
        def __init__(self):
            self.current_ascii = 97

        def yield_next_letter(self):
            next_ltr = chr(self.current_ascii)
            self.current_ascii += 1
            return next_ltr
    
    letter_gen = LetterGenerator()
    einsum_idxs = [
        "".join([
            letter_gen.yield_next_letter() for _ in range(len(S_at_pt.shape))
        ]) for S_at_pt in S_evaluated_at_points
    ]
    einsum_lhs = ",".join(einsum_idxs)
    einsum_rhs = "".join(einsum_idxs)
    einsum_str = f"{einsum_lhs}->{einsum_rhs}"

    n_point_fct = np.einsum(einsum_str, *S_evaluated_at_points)
    
    return n_point_fct