import numpy as np
from openfermion import FermionOperator, bravyi_kitaev
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
    def __init__(self, molecule, n_step, initial_circuit):
        self.initial_circuit = initial_circuit
        hamiltonian = to_pauli_expr_with_bk(molecule)
        h = molecule.get_molecular_hamiltonian()
        def inv(k):
            return tuple((x, 0 if y else 1) for x, y in reversed(k))
        ferms = [(FermionOperator(k, h[k]), FermionOperator(inv(k), h[k])) for k in h]
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
        super().__init__(hamiltonian, len(self.terms))

    def get_circuit(self, params):
        c = self.initial_circuit.copy()
        for t, terms in zip(params, self.terms):
            for term in terms:
                term.get_time_evolution()(c, t * np.pi)
        return c

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
