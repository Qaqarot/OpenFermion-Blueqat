from openfermion.hamiltonians import MolecularData
from openfermion.ops import InteractionOperator, FermionOperator, QubitOperator
from openfermion.transforms import bravyi_kitaev, get_fermion_operator
from blueqat.pauli import *

def to_pauli_expr(qubit_operator):
    """Convert OpenFermion `QubitOperator` to Blueqat `PauliExpr`."""
    def convert_ops(qo_ops, coeff):
        return Term.from_ops_iter((pauli_from_char(c, n) for n, c in qo_ops), coeff)

    return Expr.from_terms_iter(convert_ops(ops, coeff) for ops, coeff in qubit_operator.terms.items())

def from_pauli_term(term):
    """Convert Blueqat `PauliTerm` to OpenFermion `QubitOperator`.
    Note: Use `from_pauli_expr` instead of this."""
    term = term.to_term()
    def ops_to_str(bq_ops):
        s_ops = []
        for op in bq_ops:
            s_ops.append(op.op + str(op.n))
        return " ".join(s_ops)
    return QubitOperator(ops_to_str(term.ops), term.coeff)

def from_pauli_expr(expr):
    """Convert Blueqat `PauliExpr` or `PauliTerm` to OpenFermion `QubitOperator`."""
    terms = expr.to_expr().terms
    if not terms:
        return QubitOperator("I", 0)
    qo = from_pauli_term(terms[0])
    for term in terms[1:]:
        qo += from_pauli_term(term)
    return qo

def to_pauli_expr_with_bk(hamiltonian):
    """Convert from OpenFermion's `MolecularData`, `InteractionOperator`,
    `FermionOperator` or `QubitOperator` to Blueqat `PauliExpr`
    via Bravyi Kitaev transformation."""
    if isinstance(hamiltonian, MolecularData):
        hamiltonian = hamiltonian.get_molecular_hamiltonian()
    if isinstance(hamiltonian, InteractionOperator):
        hamiltonian = get_fermion_operator(hamiltonian)
    if isinstance(hamiltonian, FermionOperator):
        hamiltonian = bravyi_kitaev(hamiltonian)
    if isinstance(hamiltonian, QubitOperator):
        hamiltonian = to_pauli_expr(hamiltonian)
    return hamiltonian
