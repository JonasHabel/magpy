import numpy as np
from magpy.lattice import HoneycombLatticeA
from magpy.interactions import KitaevInteraction, UniformMagneticField
from magpy.greens_functions import *
from magpy.spectral_function import *
from magpy.structure_factors import *
from magpy.largeS import LSWT
from magpy.momenta_utils import Momenta
from magpy.util import PAULI_MATRICES
from . import test_models



def test_structure_factor_honeycomb_DMI_non_interacting():
    model, (J, D, S_A, S_B, theta) = test_models.FM_Heisenberg_with_DMI_honeycomb(S_A=1, S_B=1)
    kpath = model.lattice.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
        ["Gamma", "K", "M", "Gamma"], 50)

    eigws, eigvs = LSWT.get_eigensystems_momentum_space(model, Momenta.of(kpath))
    eigws, eigvs = eigws.raw_quantity[0], eigvs.raw_quantity[0]

    freqs = np.linspace(0, 6, 101)
    sigma_z = PAULI_MATRICES[2]
    num_bands = model.lattice.num_sites_unit_cell
    GF_non_interacting = np.array([
        np.array([
            np.diag([
                1. / (freq*ph_sign - eigws[k_idx, 2*band_idx] + 0.1j) \
                for band_idx in range(num_bands) \
                for ph_sign in (1, -1)
            ]) for freq in freqs
        ]) for k_idx, _ in enumerate(kpath.ks)
    ])

    spec_funcs = np.array([
        compute_spectral_function(GF_non_interacting[k_idx, :, ::2, ::2]) \
        for k_idx, k in enumerate(kpath.ks)
    ])

    struct_facts = np.array([
        compute_structure_factor(GF_non_interacting[k_idx], k, eigvs[k_idx], model) \
        for k_idx, k in enumerate(kpath.ks)
    ])
    
    import matplotlib.pyplot as plt
    plt.contourf(np.arange(len(kpath.ks)), freqs, struct_facts[:, :, 1, 1].T, levels=100)
    plt.colorbar()
    plt.show()



def test_structure_factor_JKGamma_non_interacting():
    # reproducing McClarty's plots (SuppMat of Topological Magnons in Kitaev Magnets ...)
    lattice = HoneycombLatticeA()
    lattice.sublattices = np.array([[0, 0.5], [0, -0.5]])
    model = Model(
        lattice,
        [
            KitaevInteraction(lattice, K=2*1.0),
            UniformMagneticField(lattice, 3.0 * np.array([1., 1., 1.])/np.sqrt(3))
        ],
        0.5 * np.array([[1, 1, 1], [1, 1, 1]]) / np.sqrt(3),
    )
    kpath = model.lattice.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
        ["Gamma'", "M", "Gamma", "K", "M'"], 50, custom_hisym_points={
            # "Gamma'": np.array([1., 1.]),
            # "M": np.array([1/2, 1/2]),
            # "Gamma": np.array([0., 0.]),
            # "K": np.array([1/3, 2/3]),
            # "M'": np.array([1/2, 1.]),
            "Gamma'": np.array([0., 1.]),
            "M": np.array([0., 1/2]),
            "Gamma": np.array([0., 0.]),
            "K": np.array([-1/3, 1/3]),
            "M'": np.array([-1/2, 1/2]),
        })

    eigws, eigvs = LSWT.get_eigensystems_momentum_space(model, Momenta.of(kpath))
    eigws, eigvs = eigws.raw_quantity[0], eigvs.raw_quantity[0]

    freqs = np.linspace(0, 4, 101)
    sigma_z = PAULI_MATRICES[2]
    num_bands = model.lattice.num_sites_unit_cell
    GF_non_interacting = np.array([
        np.array([
            np.diag([
                1. / (freq*ph_sign - eigws[k_idx, 2*band_idx] + 0.01j) \
                for band_idx in range(num_bands) \
                for ph_sign in (1, -1)
            ]) for freq in freqs
        ]) for k_idx, _ in enumerate(kpath.ks)
    ])

    spec_funcs = np.array([
        compute_spectral_function(GF_non_interacting[k_idx, :, ::2, ::2]) \
        for k_idx, k in enumerate(kpath.ks)
    ])

    struct_facts = np.array([
        compute_structure_factor(GF_non_interacting[k_idx], k, eigvs[k_idx], model) \
        for k_idx, k in enumerate(kpath.ks)
    ])
    
    import matplotlib.pyplot as plt
    from magpy.plot import plot_util
    #sf = struct_facts[:, :, 0, 0].T + struct_facts[:, :, 1, 1].T
    sf = np.log(struct_facts[:, :, 0, 0].T + struct_facts[:, :, 1, 1].T)
    sf[np.where(sf < -3)] = -3
    plt.contourf(np.arange(len(kpath.ks)), freqs, sf, levels=100)
    plt.colorbar()
    plot_util.set_momentum_path_x_ticks(plt.gca(), kpath)
    plt.show()
