from magpy.largeS import real_space


def get_real_space_magnon_Hamiltonian(
        interaction_Hamiltonian_real_space, model, order):
    return interaction_Hamiltonian_real_space \
        if interaction_Hamiltonian_real_space is not None \
        else real_space.compute_magnon_Hamiltonian(model, order)