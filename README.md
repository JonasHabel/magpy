magpy
=====

magpy is a Python package for easy numerical spin-wave computations in spin lattices.

The following code produces the linear spin-wave band structure along the Gamma-X-M-Gamma path in reciprocal space for a square-lattice spin-3/2 ferromagnet with ferromagnetic nearest-neighbor and weak anti-ferromagnetic next-nearest-neighbor Heisenberg interactions.

    import numpy as np
    from magpy.models import Model
    from magpy.lattice import SquareLattice
    from magpy.interactions import NthNearestNeighborHeisenbergInteraction
    from magpy import LSWT
    from magpy.plot import LSWT_plot

    lattice = SquareLattice()
    model = Model(lattice, interactions=[
        NthNearestNeighborHeisenbergInteraction(lattice, n=1, J=-1.0),
        NthNearestNeighborHeisenbergInteraction(lattice, n=2, J=0.1),
    ], classical_ground_state=3/2 * np.array([[0, 0, 1]]))

    momentum_path = \
        lattice.reciprocal_lattice.get_momentum_path_approx_equally_spaced(
            ["Gamma", "X", "M", "Gamma"], 50)

    eigws, eigvs = LSWT.get_eigensystem_along_momentum_path(model, momentum_path)
    LSWT_plot.plot_energies_along_momentum_path(momentum_path, eigws)

Features
--------

- Compute magnon band structures within linear spin-wave theory
- Compute one- and two-magnon densities of state
- Compute two-magnon bubble self-energies based on cubic magnon-magnon interactions (only fully implemented for the >/>>/> bubble)
- Compute one-magnon spectral functions within both linear and non-linear spin-wave theory
- Run semi-classical Landau-Lifshitz spin dynamics simulations
- Support for arbitrary Bravais lattices with periodic or customizable open boundary conditions
- Support for arbitrary one- and two-spin interactions

Installation
------------

1. git clone the repository.
2. In the terminal, navigate to the root folder of the repository.
3. Type <code>pip install .</code>

