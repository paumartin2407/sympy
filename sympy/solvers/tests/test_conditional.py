from __future__ import annotations

import pytest

from sympy import (Eq, FiniteSet, Ne, S, Symbol, Union, solveset, symbols)
from sympy.solvers.conditional import (
    _conditional_union, _linear_polynomial_solution_branches,
    _linear_solution_branches, _quadratic_polynomial_solution_branches,
    _quadratic_solution_branches, _specialize_branches,
    _specialize_conditional_union)


def test_linear_solution_branches():
    a, b, x = symbols('a b x')
    assert _linear_solution_branches(a, b, x) == (
        (FiniteSet(-b/a), Ne(a, 0)),
        (S.Complexes, Eq(a, 0) & Eq(b, 0)),
        (S.EmptySet, Eq(a, 0) & Ne(b, 0)),
    )


def test_linear_solution_specialization_matrix():
    a, b, x = symbols('a b x')
    expr = a*x + b
    branches = _linear_polynomial_solution_branches(expr, x)
    cases = (
        ({a: 2, b: 4}, FiniteSet(-2)),
        ({a: 0, b: 0}, S.Complexes),
        ({a: 0, b: 1}, S.EmptySet),
    )
    for substitutions, expected in cases:
        actual = _specialize_conditional_union(branches, x, substitutions)
        reference = solveset(expr.xreplace(substitutions), x, S.Complexes)
        assert actual == expected == reference


def test_linear_solution_partial_and_reordered_specialization():
    a, b, x = symbols('a b x')
    branches = _linear_solution_branches(a, b, x)
    for first, second in (({b: 0}, {a: 0}), ({a: 0}, {b: 0})):
        partial = _specialize_branches(branches, first)
        assert _specialize_conditional_union(partial, x, second) == S.Complexes
    assert _specialize_conditional_union(
        branches, x, {b: 0, a: 0}) == S.Complexes


def test_linear_solution_expression_coefficients():
    p, q, r, x = symbols('p q r x')
    expr = (p - q)*x + r*p*x + p*p**2*x + q
    branches = _linear_polynomial_solution_branches(expr, x)
    assert _specialize_conditional_union(
        branches, x, {p: 0, q: 0}) == solveset(expr.xreplace(
            {p: 0, q: 0}), x, S.Complexes) == S.Complexes
    assert _specialize_conditional_union(
        branches, x, {p: 1, q: 1, r: -1}) == solveset(
            expr.xreplace({p: 1, q: 1, r: -1}), x, S.Complexes) == S.EmptySet


def test_linear_solution_constants_and_assumptions():
    x = Symbol('x')
    c = Symbol('c')
    anz = Symbol('anz', nonzero=True)

    assert _linear_solution_branches(2, 1, x) == (
        (FiniteSet(S.Half * -1), S.true),)
    assert _linear_polynomial_solution_branches(0, x) == ((S.Complexes, S.true),)
    assert _linear_polynomial_solution_branches(3, x) == ((S.EmptySet, S.true),)
    assert _specialize_conditional_union(
        _linear_polynomial_solution_branches(c, x), x, {c: 0}) == S.Complexes
    assert _specialize_conditional_union(
        _linear_polynomial_solution_branches(c, x), x, {c: 3}) == S.EmptySet
    assert _linear_solution_branches(anz, c, x) == (
        (FiniteSet(-c/anz), S.true),)


def test_linear_polynomial_rejects_higher_degree():
    x, c = symbols('x c')
    with pytest.raises(ValueError, match='degree at most one'):
        _linear_polynomial_solution_branches(x**2 + 1, x)
    with pytest.raises(ValueError, match='degree at most one'):
        _linear_polynomial_solution_branches(c*x**2 + 1, x)


def test_linear_solution_coefficients_cannot_contain_solve_variable():
    x = Symbol('x')
    with pytest.raises(ValueError, match='must not contain the solve variable'):
        _linear_solution_branches(x, 1, x)


def test_quadratic_solution_degree_degeneration_matrix():
    a, b, c, x = symbols('a b c x')
    expr = a*x**2 + b*x + c
    branches = _quadratic_polynomial_solution_branches(expr, x)
    cases = (
        ({a: 1, b: 0, c: -1}, FiniteSet(-1, 1)),
        ({a: 1, b: 2, c: 1}, FiniteSet(-1)),
        ({a: 1, b: 0, c: 1}, FiniteSet(-S.ImaginaryUnit, S.ImaginaryUnit)),
        ({a: 0, b: 2, c: 4}, FiniteSet(-2)),
        ({a: 0, b: 0, c: 0}, S.Complexes),
        ({a: 0, b: 0, c: 1}, S.EmptySet),
    )
    for substitutions, expected in cases:
        actual = _specialize_conditional_union(branches, x, substitutions)
        reference = solveset(expr.xreplace(substitutions), x, S.Complexes)
        assert actual == expected == reference

    assert _specialize_conditional_union(branches, x, {a: 0}) == \
        _conditional_union(_linear_solution_branches(b, c, x), x)
    assert _specialize_conditional_union(
        branches, x, {a: 0, b: 0, c: 0}) is S.Complexes


def test_quadratic_solution_symbolic_and_assumed_coefficients():
    a, b, c, x = symbols('a b c x')
    branches = _quadratic_solution_branches(a, b, c, x)
    assert branches[0][1] == Ne(a, 0)
    assert branches[1][1] == (Eq(a, 0) & Ne(b, 0))
    assert branches[2][1] == (Eq(a, 0) & Eq(b, 0) & Eq(c, 0))
    assert branches[3][1] == (Eq(a, 0) & Eq(b, 0) & Ne(c, 0))

    anz = Symbol('anz', nonzero=True)
    assumed = _quadratic_solution_branches(anz, b, c, x)
    assert len(assumed) == 1
    assert assumed[0][1] is S.true


def test_quadratic_polynomial_rejects_higher_degree():
    x, d = symbols('x d')
    with pytest.raises(ValueError, match='degree at most two'):
        _quadratic_polynomial_solution_branches(x**3 + 1, x)
    with pytest.raises(ValueError, match='degree at most two'):
        _quadratic_polynomial_solution_branches(d*x**3 + 1, x)


def test_quadratic_zero_and_constant_polynomials():
    x, c = symbols('x c')
    assert _quadratic_polynomial_solution_branches(0, x) == (
        (S.Complexes, S.true),)
    assert _quadratic_polynomial_solution_branches(3, x) == (
        (S.EmptySet, S.true),)
    branches = _quadratic_polynomial_solution_branches(c, x)
    assert _specialize_conditional_union(branches, x, {c: 0}) == S.Complexes
    assert _specialize_conditional_union(branches, x, {c: 3}) == S.EmptySet


def test_quadratic_dependent_coefficients_specialization():
    p, q, x = symbols('p q x')
    expr = p*x**2 + p*x + q
    branches = _quadratic_polynomial_solution_branches(expr, x)

    assert _specialize_conditional_union(
        branches, x, {p: 0, q: 0}) == S.Complexes
    assert _specialize_conditional_union(
        branches, x, {p: 0, q: 1}) == S.EmptySet

    partial = _specialize_conditional_union(branches, x, {p: 0})
    assert partial == _conditional_union(
        _linear_solution_branches(0, q, x), x)
    assert _specialize_conditional_union(
        _specialize_branches(branches, {p: 0}), x, {q: 0}) == S.Complexes
    assert _specialize_conditional_union(
        _specialize_branches(branches, {q: 0}), x, {p: 0}) == S.Complexes
    assert _specialize_conditional_union(
        _specialize_branches(branches, {p: 0}), x, {q: 1}) == S.EmptySet
    assert _specialize_conditional_union(
        _specialize_branches(branches, {q: 1}), x, {p: 0}) == S.EmptySet


def test_guarded_condition_set_union_representation():
    a, b, x = symbols('a b x')
    guarded = _conditional_union(_linear_solution_branches(a, b, x), x)
    assert guarded.xreplace({a: 2, b: 4}) == FiniteSet(-2)
    assert guarded.xreplace({a: 0, b: 0}) == S.Complexes
    assert guarded.xreplace({a: 0, b: 1}) == S.EmptySet
    assert isinstance(guarded, Union)
