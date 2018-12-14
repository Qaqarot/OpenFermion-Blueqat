import numpy as np
from blueqat import Circuit
from blueqat.vqe import AnsatzBase
from ._transform import to_pauli_expr_with_bk
from .bk import to_bk_basis

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

def get_bk_initialize_circuit(indices, n_qubits=None):
    """This function is experimental and temporary.

    Get the circuit to make initial state.
    Qubits are encoded by Bravyi Kitaev transformation."""
    return Circuit().x[tuple(sorted(to_bk_basis(indices, n_qubits)))]

def get_hf_circuit(molecule):
    return get_bk_initialize_circuit(range(molecule.n_electrons), molecule.n_qubits)
