import numpy as np
from magpy.momenta_utils import CollapseMomenta, Momenta, RestoreMomenta, Target


def test_momenta():
    k_path = np.linspace(np.array([0, 0]), np.array([1, 2]), 10)
    k_BZ = np.array([*np.meshgrid(np.linspace(0, 1, 5), np.linspace(-1, 2, 30))]).T

    k_arrays = [k_path, k_BZ]
    momenta = Momenta(*k_arrays)

    collapsed_momenta = momenta.collapse(momenta)
    expected_collapsed_momenta = [
        k_path,
        k_BZ.reshape((5*30, 2))
    ]
    for k, exp_k in zip(collapsed_momenta, expected_collapsed_momenta):
        assert np.allclose(k, exp_k)

    restored_momenta = momenta.restore(collapsed_momenta)
    expected_restored_momenta = k_arrays

    for k, exp_k in zip(restored_momenta, expected_restored_momenta):
        assert np.allclose(k, exp_k)


def test_eigenvectors():
    k_path = np.linspace(np.array([0, 0]), np.array([1, 2]), 10)
    k_BZ = np.array([*np.meshgrid(np.linspace(0, 1, 5), np.linspace(-1, 2, 15))]).T

    k_arrays = [k_path, k_BZ]
    momenta = Momenta(*k_arrays)

    np.random.seed(1)
    eigvs = [
        np.random.rand(10, 2, 2),
        np.random.rand(5, 15, 2, 2),
    ]

    collapsed_eigvs = momenta.collapse(eigvs)
    expected_collapsed_eigvs = [
        eigvs[0],
        eigvs[1].reshape((5*15, 2, 2))
    ]
    for ev, exp_ev in zip(collapsed_eigvs, expected_collapsed_eigvs):
        assert np.allclose(ev, exp_ev)

    restored_eigvs = momenta.restore(collapsed_eigvs)
    expected_restored_eigvs = eigvs

    for ev, exp_ev in zip(restored_eigvs, expected_restored_eigvs):
        assert np.allclose(ev, exp_ev)


def test_permuted_magnon_Hamiltonians():
    k_path = np.linspace(np.array([0, 0]), np.array([1, 2]), 10)
    k_BZ = np.array([*np.meshgrid(np.linspace(0, 1, 5), np.linspace(-1, 2, 15))]).T

    k_arrays = [k_path, k_BZ]
    momenta = Momenta(*k_arrays)

    np.random.seed(1)
    magnon_Hs = np.random.rand(6, 10, 5, 15, 2, 2, 2)

    # test non-deep collapse
    collapsed_magnon_Hs = momenta.collapse_tensor(magnon_Hs, first_momentum_idx=1, deep=False)
    expected_collapsed_magnon_Hs = magnon_Hs.reshape((6, 10, 75, 2, 2, 2))
    assert np.allclose(collapsed_magnon_Hs, expected_collapsed_magnon_Hs)

    restored_magnon_Hs = momenta.restore_tensor(collapsed_magnon_Hs, first_momentum_idx=1, deep=False)
    assert np.allclose(restored_magnon_Hs, magnon_Hs)

    # test deep collapse
    deep_collapsed_magnon_Hs = momenta.collapse_tensor(magnon_Hs, first_momentum_idx=1, deep=True)
    expected_deep_collapsed_magnon_Hs = magnon_Hs.reshape((6, 750, 2, 2, 2))
    assert np.allclose(deep_collapsed_magnon_Hs, expected_deep_collapsed_magnon_Hs)

    deep_restored_magnon_Hs = momenta.restore_tensor(deep_collapsed_magnon_Hs, first_momentum_idx=1, deep=True)
    assert np.allclose(deep_restored_magnon_Hs, magnon_Hs)




def test_decorator():
    k_path = np.linspace(np.array([0, 0]), np.array([1, 2]), 10)
    k_BZ = np.array([*np.meshgrid(np.linspace(0, 1, 5), np.linspace(-1, 2, 15))]).T

    k_arrays = [k_path, k_BZ]
    momenta = Momenta(*k_arrays)

    np.random.seed(1)
    eigvs = [
        np.random.rand(10, 2, 2),
        np.random.rand(5, 15, 2, 2),
    ]
    magnon_Hs = np.random.rand(6, 10, 5, 15, 2, 2, 2)

    @RestoreMomenta(
        momentum_arrays_arg_idx=1,
        output_first_momentum_idx=1,
        output_is_tensor=True,
        output_restore_deep=True,
    )
    @CollapseMomenta(
        momentum_arrays_arg_idx=1,
        targets=(
            Target(arg_idx=1, first_momentum_idx=0, is_tensor=False),
            Target(arg_idx=2, first_momentum_idx=0, is_tensor=False),
            Target(arg_idx=3, first_momentum_idx=1, is_tensor=True),
            Target(arg_idx=4, first_momentum_idx=1, is_tensor=True, collapse_deep=True),
        )
    )
    def some_function(model, k_arrays, eigvs, magnon_Hs, magnon_Hs_deep, *args):
        assert k_arrays[0].shape == (10, 2)
        assert k_arrays[1].shape == (5*15, 2)
        assert eigvs[0].shape == (10, 2, 2)
        assert eigvs[1].shape == (5*15, 2, 2)
        assert magnon_Hs.shape == (6, 10, 5*15, 2, 2, 2)
        assert magnon_Hs_deep.shape == (6, 10*5*15, 2, 2, 2)
        return len(magnon_Hs_deep.shape) * magnon_Hs_deep
    
    result = some_function(None, momenta, eigvs, magnon_Hs, magnon_Hs)

    assert np.allclose(result, 5*magnon_Hs)
    


    
