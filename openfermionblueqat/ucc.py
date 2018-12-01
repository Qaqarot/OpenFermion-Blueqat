from itertools import cycle
from functools import reduce
import numpy as np
from openfermion import FermionOperator, bravyi_kitaev, get_fermion_operator
from blueqat import Circuit, pauli
from blueqat.vqe import AnsatzBase
from ._transform import to_pauli_expr_with_bk, to_pauli_expr
#from .bk import get_update_set

class UCCAnsatz(AnsatzBase):
    """Ansatz of Unitary Coupled Cluster."""
    def __init__(self, hamiltonian, n_step, initial_circuit):
        self.initial_circuit = initial_circuit
        hamiltonian = to_pauli_expr_with_bk(hamiltonian)
        super().__init__(hamiltonian, n_step)

    def get_circuit(self, params):
        c = self.initial_circuit.copy()
        for t in params:
            for term in self.hamiltonian.terms:
                term.get_time_evolution()(c, t * np.pi)
        return c

class UCCAnsatz2(AnsatzBase):
    """Ansatz of Unitary Coupled Cluster."""
    def __init__(self, molecule, initial_circuit, n_params=None):
        self.initial_circuit = initial_circuit
        hamiltonian = to_pauli_expr_with_bk(molecule)
        h = get_fermion_operator(molecule.get_molecular_hamiltonian())
        trim_zero_fermion_operator(h)
        trim_duplicated_fermion_operator(h)
        trim_conjugate_fermion_operator(h)
        def inv(k):
            return tuple((x, 0 if y else 1) for x, y in reversed(k))
        ferms = [(FermionOperator(k, h.terms[k]), FermionOperator(inv(k), h.terms[k])) for k in h.terms]
        qubits = [to_pauli_expr(bravyi_kitaev(t - tdg)) * 1.j for t, tdg in ferms]
        zero = pauli.Expr.zero()
        qubits = [q for q in qubits if q != zero]
        #print(qubits)
        self.terms = []
        for terms in qubits:
            a = []
            for term in terms:
                a.append(term)
            self.terms.append(a)
        #print(self.terms)
        if n_params is None:
            n_params = len(self.terms)
        elif 0 < n_params < 1.0:
            n_params = int(len(self.terms) * n_params)
        if not isinstance(n_params, int):
            raise ValueError("n_params shall be None, 0.0 < n_params < 1.0, or integer")
        if n_params > len(self.terms):
            n_params = self.terms
        super().__init__(hamiltonian, n_params)

    def get_circuit(self, params):
        c = self.initial_circuit.copy()
        for t, terms in zip(cycle(params), self.terms):
            for term in terms:
                term.get_time_evolution()(c, t * np.pi)
        return c

class UCCAnsatz3(AnsatzBase):
    """Ansatz of Unitary Coupled Cluster."""
    def __init__(self, molecule, n_step, initial_circuit):
        self.initial_circuit = initial_circuit
        hamiltonian = to_pauli_expr_with_bk(molecule)
        h = get_fermion_operator(molecule.get_molecular_hamiltonian())
        trim_zero_fermion_operator(h)
        trim_duplicated_fermion_operator(h)
        trim_conjugate_fermion_operator(h)
        def inv(k):
            return tuple((x, 0 if y else 1) for x, y in reversed(k))
        ferms = [(FermionOperator(k, h.terms[k]), FermionOperator(inv(k), h.terms[k])) for k in h.terms]
        qubits = [to_pauli_expr(bravyi_kitaev(t - tdg)) * 1.j for t, tdg in ferms]
        zero = pauli.Expr.zero()
        qubits = [q for q in qubits if q != zero]
        #print(qubits)
        self.terms = []
        for terms in qubits:
            a = []
            for term in terms:
                a.append(term)
            self.terms.append(a)
        #print(self.terms)
        self.n_qubits = molecule.n_orbitals * 2
        self.n_step = n_step
        super().__init__(hamiltonian, self.n_qubits * n_step)

    def get_circuit(self, params):
        c = self.initial_circuit.copy()
        p_ofs = 0
        for _ in range(self.n_step):
            for terms in self.terms:
                for term in terms:
                    t = sum(params[p_ofs + x] for x in term.n_iter()) / len(term)
                    term.get_time_evolution()(c, t * np.pi)
            p_ofs += self.n_qubits
        return c

class UCCAnsatz4(AnsatzBase):
    """Ansatz of Unitary Coupled Cluster."""
    def __init__(self, molecule, n_step, initial_circuit):
        self.initial_circuit = initial_circuit
        hamiltonian = to_pauli_expr_with_bk(molecule)
        h = get_fermion_operator(molecule.get_molecular_hamiltonian())
        trim_zero_fermion_operator(h)
        trim_duplicated_fermion_operator(h)
        trim_conjugate_fermion_operator(h)
        def inv(k):
            return tuple((x, 0 if y else 1) for x, y in reversed(k))
        fermis = [(FermionOperator(k, h.terms[k]), FermionOperator(inv(k), h.terms[k])) for k in h.terms]
        qubits = [to_pauli_expr(bravyi_kitaev(t - tdg)) * 1.j for t, tdg in fermis]
        zero = pauli.Expr.zero()
        qubits = [q for q in qubits if q != zero]
        #print(qubits)
        self.sterms = []
        self.dterms = []
        for q in qubits:
            a = []
            for term in q:
                a.append(term)
            if len(a) == 4:
                self.dterms.append(a)
            if len(a) == 2:
                self.sterms.append(a)
        #print(self.terms)
        self.n_step = n_step
        super().__init__(hamiltonian, n_step * 2)

    def get_circuit(self, params):
        c = self.initial_circuit.copy()
        p_ofs = 0
        for t2, t4 in zip(params, params[self.n_step:]):
            for terms in self.sterms:
                for term in terms:
                    term.get_time_evolution()(c, t2 * np.pi)
            for terms in self.dterms:
                for term in terms:
                    term.get_time_evolution()(c, t4 * np.pi)
        return c

def trim_zero_fermion_operator(fermion):
    """Remove [i^ i^ a b] or [i^ j^ a a]"""
    keys = list(fermion.terms)
    for k in keys:
        if not k:
            continue
        before = k[0]
        for t in k[1:]:
            if before == t:
                del fermion.terms[k]
                break
            before = t

def trim_duplicated_fermion_operator(fermion):
    """Remove [j^ i^ b a] if exists [i^ j^ a b]"""
    keys = list(fermion.terms)
    for k in keys:
        if len(k) != 4:
            continue
        if k[0][1] != 1 or k[1][1] != 1 or k[2][1] != 0 or k[3][1] != 0:
            continue
        i, j, a, b = k[0][0], k[1][0], k[2][0], k[3][0]
        if i > j or (i == j and a > b):
            k2 = ((j, 1), (i, 1), (b, 0), (a, 0))
            if k2 in keys and k2 != k:
                # These values must be almost same.
                assert abs(fermion.terms[k] - fermion.terms[k2]) < 0.000001
                del fermion.terms[k2]

def trim_conjugate_fermion_operator(fermion):
    """Remove [b^ a^ j i], [a^ b^ i j] or [a^ i] if exists [i^ j^ a b] or [i^ a]"""
    keys = list(fermion.terms)
    for k in keys:
        if len(k) == 2:
            if k[0][1] != 1 or k[1][1] != 0:
                continue
            i, a = k[0][0], k[1][0]
            if i > a:
                k2 = ((a, 1), (i, 0))
                if k2 in keys and k2 != k:
                    # A complex conjugate of k2's value must be almost as same as k's value.
                    assert abs(fermion.terms[k] - fermion.terms[k2].conjugate()) < 0.000001
                    del fermion.terms[k2]
        if len(k) == 4:
            if k[0][1] != 1 or k[1][1] != 1 or k[2][1] != 0 or k[3][1] != 0:
                continue
            i, j, a, b = k[0][0], k[1][0], k[2][0], k[3][0]
            k2 = ((b, 1), (a, 1), (j, 0), (i, 0))
            if i > b or (i == b and j > a):
                if k2 in keys and k2 != k:
                    # A complex conjugate of k2's value must be almost as same as k's value.
                    assert abs(fermion.terms[k] - fermion.terms[k2].conjugate()) < 0.000001
                    del fermion.terms[k2]
            k3 = ((a, 1), (b, 1), (i, 0), (j, 0))
            if i > a or (i == a and j > b):
                if k3 in keys and k3 != k and k3 != k2:
                    # A complex conjugate of k2's value must be almost as same as k's value.
                    assert abs(fermion.terms[k] - fermion.terms[k3].conjugate()) < 0.000001
                    del fermion.terms[k3]
'''
def get_bk_initialize_circuit(elecs, n_qubits=None):
    """This function is experimental and temporary.

    Get the circuit to make initial state.
    Qubits are encoded by Bravyi Kitaev transformation."""
    elecs = tuple(elecs)
    if n_qubits is None:
        n_qubits = max(elecs) + 1
    bits = set()
    for e in elecs:
        bits ^= get_update_set(e, n_qubits)
    return Circuit().x[tuple(bits)]
'''
