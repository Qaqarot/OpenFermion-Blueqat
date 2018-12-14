from openfermionblueqat.bk import *

def test_bk_basis1():
    assert to_bk_basis((0, 1)) == set([0])

def test_bk_basis2():
    assert to_bk_basis(range(4)) == set([0, 2])

def test_bk_basis3():
    assert to_bk_basis((0,), 3) == set([0, 1])
    assert to_bk_basis((0,), 4) == set([0, 1, 3])
    assert to_bk_basis((0,), 7) == set([0, 1, 3])
    assert to_bk_basis((0,), 8) == set([0, 1, 3, 7])
    assert to_bk_basis((0,), 9) == set([0, 1, 3, 7])
