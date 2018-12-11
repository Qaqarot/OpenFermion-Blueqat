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

def ucc_t1(r, a, n_qubits):
    """Returns ([r^ a] - [r^ a]†) operator in BK basis."""
    an, cr = a_adg_pair(n_qubits)
    return (1j * (cr(r) * an(a) - cr(a) * an(r))).simplify()
    # annihilation(a) = 0.5 (X_U(a) X_a Z_P(a) + j X_U(a) Y_a Z_ρ(a))
    # creation(r)     = 0.5 (X_U(r) X_r Z_P(r) - j X_U(r) Y_r Z_ρ(r))
    # Therefore,
    # [r^ a]  = 0.25 (X_U(r) X_r Z_P(r) X_U(a) X_a Z_P(a)
    #              -  X_U(r) Y_r Z_ρ(r) X_U(a) Y_a Z_ρ(a)
    #              +j X_U(r) X_r Z_P(r) X_U(a) Y_a Z_ρ(a)
    #              -j X_U(r) Y_r Z_ρ(r) X_U(a) X_a Z_P(a))
    # [r^ a]† = 0.25 (X_U(r) X_r Z_P(r) X_U(a) X_a Z_P(a)
    #              -  X_U(r) Y_r Z_ρ(r) X_U(a) Y_a Z_ρ(a)
    #              -j X_U(r) X_r Z_P(r) X_U(a) Y_a Z_ρ(a)
    #              +j X_U(r) Y_r Z_ρ(r) X_U(a) X_a Z_P(a))
    # [r^ a] - [r^ a]†
    #         = 0.5j (X_U(r) X_r Z_P(r) X_U(a) Y_a Z_ρ(a)
    #               - X_U(r) Y_r Z_ρ(r) X_U(a) X_a Z_P(a))
    # This function returns [r^ a] - [r^ a]† but eliminate coefficient 0.5j.
    prod = pauli_product
    ur = get_update_set(r, n_qubits, False)
    ua = get_update_set(a, n_qubits, False)
    pr = get_parity_set(r)
    pa = get_parity_set(a)
    fr = get_flip_set(r)
    rr = pr - fr
    fa = get_flip_set(a)
    ra = pa - fa
    return (prod(X, ur) * X[r] * prod(Z, pr) * prod(X, ua) * Y[a] * prod(Z, ra) -
            prod(X, ur) * Y[r] * prod(Z, rr) * prod(X, ua) * X[a] * prod(Z, pa)).simplify()

def ucc_t2(r, s, a, b, n_qubits):
    """Returns ([r^ s^ b a] - [r^ s^ b a]†) operator in BK basis."""
    an, cr = a_adg_pair(n_qubits)
    return (1j * (cr(r) * cr(s) * an(b) * an(a) - cr(a) * cr(b) * an(s) * an(r))).simplify()
    # Assume that
    # annihilation(a) = X(a) + jY(a)
    # creation(r)     = X(r) - jY(r)
    # Then,
    # [r^ s^ b a] =  (X(r) - jY(r))(X(s) - jY(s))(X(b) + jY(b))(X(a) + jY(a))
    #             =  [(X(r)X(s) - Y(r)Y(s))(X(b)X(a) - Y(b)Y(a))
    #              +  (Y(r)X(s) + X(r)Y(s))(Y(b)X(a) + X(b)Y(a))]
    #              +j[(X(r)X(s) - Y(r)Y(s))(Y(b)X(a) + X(b)Y(a))
    #              -  (Y(r)X(s) + X(r)Y(s))(X(b)X(a) - Y(b)Y(a))]
    # [r^ s^ b a] - [r^ s^ b a]† preserved only imaginary part.
    # This operator is not applied Bravyi-Kitaev basis,
    # however we can represent it in Bravyi-Kitaev basis using
    # BK transformed annihilation and creation operator.
    # annihilation(a) = 0.5 (X_U(a) X_a Z_P(a) + j X_U(a) Y_a Z_ρ(a))
    # creation(r)     = 0.5 (X_U(r) X_r Z_P(r) - j X_U(r) Y_r Z_ρ(r))
    ur = get_update_set(r, n_qubits, False)
    us = get_update_set(s, n_qubits, False)
    ua = get_update_set(a, n_qubits, False)
    ub = get_update_set(b, n_qubits, False)
    pr = get_parity_set(r)
    ps = get_parity_set(s)
    pa = get_parity_set(a)
    pb = get_parity_set(b)
    rr = pr - get_flip_set(r)
    rs = ps - get_flip_set(s)
    ra = pa - get_flip_set(a)
    rb = pb - get_flip_set(b)
    xr = pauli_product(X, ur) * X[r] * pauli_product(Z, pr)
    yr = pauli_product(X, ur) * Y[r] * pauli_product(Z, rr)
    xs = pauli_product(X, us) * X[s] * pauli_product(Z, ps)
    ys = pauli_product(X, us) * Y[s] * pauli_product(Z, rs)
    xa = pauli_product(X, ua) * X[a] * pauli_product(Z, pa)
    ya = pauli_product(X, ua) * Y[a] * pauli_product(Z, ra)
    xb = pauli_product(X, ub) * X[b] * pauli_product(Z, pb)
    yb = pauli_product(X, ub) * Y[b] * pauli_product(Z, rb)
    return ((xr * xs - yr * ys) * (yb * xa + xb * ya) -
            (yr * xs + xr * ys) * (xb * xa - yb * ya)).simplify()
