import numpy as np
from blueqat import Circuit
from blueqat.vqe import AnsatzBase
from ._transform import to_pauli_expr_with_bk

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

def get_bk_initialize_circuit(hamiltonian):
    """This function is experimental and temporary.

    Get the circuit to make initial state.
    Qubits are encoded by Bravyi Kitaev transformation."""
    hamiltonian = to_pauli_expr_with_bk(hamiltonian)
    mat = hamiltonian.to_matrix()
    state = mat[:, 0]
    assert all(abs(x) < 0.0001 or abs(x - 1) < 0.0001 for x in state)
    bits = tuple(i for i, v in enumerate(state) if abs(v - 1) < 0.0001)
    c = Circuit(len(state) + 1)
    return c.x[bits]
