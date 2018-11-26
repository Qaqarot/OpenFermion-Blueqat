import numpy as np
from blueqat import Circuit
from blueqat.vqe import AnsatzBase
from ._transform import to_pauli_expr_with_bk
from .bk import get_update_set

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
