from functools import reduce
import numpy as np
from blueqat.pauli import I, X, Y, Z

def make_bravyi_kitaev_matrix(n_qubits):
    if n_orbitals <= 1:
        return np.array([[1]])
    n = 2
    while n < n_orbitals:
        n *= 2
    arr = np.zeros((n, n), dtype=np.int8)
    arr[0,0] = arr[1,0] = arr[1,1] = 1
    m = 2
    while m < n:
        arr[m:m+m, m:m+m] = arr[:m, :m]
        arr[m+m-1,:m] = 1
        m += m
    return arr

def make_bravyi_kitaev_inv_matrix(n_qubits):
    if n_orbitals <= 1:
        return np.array([[1]])
    n = 2
    while n < n_orbitals:
        n *= 2
    arr = np.zeros((n, n), dtype=np.int8)
    arr[0,0] = arr[1,0] = arr[1,1] = 1
    m = 2
    while m < n:
        arr[m:m+m, m:m+m] = arr[:m, :m]
        arr[m+m-1,m-1] = 1
        m += m
    return arr

def get_update_set(index, n_qubits, include_index=True):
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
    if include_index:
        u.add(index)
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

def pauli_product(cls, targets):
    return reduce(lambda pauli, target: pauli * cls(target), targets, I)

def annihilation(index, n_qubits):
    x_u = pauli_product(X, get_update_set(index, n_qubits, False))
    z_p = pauli_product(Z, get_parity_set(index))
    if index % 2 == 1:
        z_r = pauli_product(Z, get_parity_set(index) - get_flip_set(index))
        return x_u * X[index] * z_p * 0.5 + x_u * Y[index] * z_r * 0.5j
    return x_u * X[index] * z_p * 0.5 + x_u * Y[index] * z_p * 0.5j

def creation(index, n_qubits):
    x_u = pauli_product(X, get_update_set(index, n_qubits, False))
    z_p = pauli_product(Z, get_parity_set(index))
    if index % 2 == 1:
        z_r = pauli_product(Z, get_parity_set(index) - get_flip_set(index))
        return x_u * X[index] * z_p * 0.5 + x_u * Y[index] * z_r * -0.5j
    return x_u * X[index] * z_p * 0.5 + x_u * Y[index] * z_p * -0.5j

def a_adg_pair(n_qubits):
    return (lambda i: annihilation(i, n_qubits)), (lambda i: creation(i, n_qubits))
