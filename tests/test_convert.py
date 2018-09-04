from openfermion import *
from openfermionblueqat import *

def test_openfermion_to_blueqat_to_openfermion():
    fermion_operator = FermionOperator("4^ 3^ 9 1", 1. + 2.j) + FermionOperator("3^ 1", -1.7)
    qubit_operator = bravyi_kitaev(fermion_operator)
    blueqat_pauli = to_pauli_expr(qubit_operator)
    qubit_operator2 = from_pauli_expr(blueqat_pauli)

    assert qubit_operator == qubit_operator2
