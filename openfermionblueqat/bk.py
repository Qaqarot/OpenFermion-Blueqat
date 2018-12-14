from functools import reduce
import numpy as np
from blueqat.pauli import I, X, Y, Z

def make_bravyi_kitaev_matrix(n_qubits):
    if n_qubits <= 1:
        return np.array([[1]])
    n = 2
    while n < n_qubits:
        n *= 2
    arr = np.zeros((n, n), dtype=np.int8)
    arr[0, 0] = arr[1, 0] = arr[1, 1] = 1
    m = 2
    while m < n:
        arr[m:m+m, m:m+m] = arr[:m, :m]
        arr[m+m-1, :m] = 1
        m += m
    return arr

def make_bravyi_kitaev_inv_matrix(n_qubits):
    if n_qubits <= 1:
        return np.array([[1]])
    n = 2
    while n < n_qubits:
        n *= 2
    arr = np.zeros((n, n), dtype=np.int8)
    arr[0, 0] = arr[1, 0] = arr[1, 1] = 1
    m = 2
    while m < n:
        arr[m:m+m, m:m+m] = arr[:m, :m]
        arr[m+m-1, m-1] = 1
        m += m
    return arr

def get_update_set(index, n_qubits):
    """Make update set"""
    if index >= n_qubits:
        raise ValueError("`index` < `n_qubits` is required.")

    n = 1
    while n < n_qubits:
        n *= 2

    def get(n, j):
        if n <= 1:
            return set()
        n_half = n // 2
        if j < n_half:
            u = get(n_half, j)
            u.add(n - 1)
            return u
        return {idx + n_half for idx in get(n_half, j - n_half)}

    u = {idx for idx in get(n, index) if idx < n_qubits}
    return u

def get_parity_set(index):
    """Make parity set"""
    n = 1
    while n <= index:
        n *= 2

    def get(n, j):
        if j <= 1:
            return {b for b in range(j)}
        n_half = n // 2
        if j < n_half:
            return get(n_half, j)
        p = {b + n_half for b in get(n_half, j - n_half)}
        p.add(n_half - 1)
        return p

    return get(n, index)

def get_flip_set(index):
    """Make flip set"""
    n = 1
    while n <= index:
        n *= 2

    def get(n, j):
        if j <= 1:
            return {b for b in range(j)}
        n_half = n // 2
        if j < n_half:
            return get(n_half, j)
        f = {b + n_half for b in get(n_half, j - n_half)}
        if j == n - 1:
            f.add(n_half - 1)
        return f

    return get(n, index)

def _pauli_product(cls, targets):
    return reduce(lambda pauli, target: pauli * cls(target), targets, I)

def annihilation(index, n_qubits):
    x_u = _pauli_product(X, get_update_set(index, n_qubits))
    z_p = _pauli_product(Z, get_parity_set(index))
    if index % 2 == 1:
        z_r = _pauli_product(Z, get_parity_set(index) - get_flip_set(index))
        return x_u * X[index] * z_p * 0.5 + x_u * Y[index] * z_r * 0.5j
    return x_u * X[index] * z_p * 0.5 + x_u * Y[index] * z_p * 0.5j

def creation(index, n_qubits):
    x_u = _pauli_product(X, get_update_set(index, n_qubits))
    z_p = _pauli_product(Z, get_parity_set(index))
    if index % 2 == 1:
        z_r = _pauli_product(Z, get_parity_set(index) - get_flip_set(index))
        return x_u * X[index] * z_p * 0.5 + x_u * Y[index] * z_r * -0.5j
    return x_u * X[index] * z_p * 0.5 + x_u * Y[index] * z_p * -0.5j

def to_bk_basis(indices, n_qubits=None):
    indices = tuple(indices)
    if n_qubits is None:
        n_qubits = max(indices) + 1
    return reduce(lambda s, i: s ^ get_update_set(i, n_qubits), indices, set(indices))
