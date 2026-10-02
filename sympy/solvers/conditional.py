"""Experimental helpers for solving parameterized polynomial equations."""
from __future__ import annotations

from sympy.core.sympify import sympify
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.logic.boolalg import And
from sympy.polys import Poly
from sympy.sets import ConditionSet, FiniteSet, S, Union


def _linear_solution_branches(a, b, x):
    """Return guarded solution sets for ``a*x + b = 0`` over Complexes.

    This is an experimental representation helper.  Each result is a pair
    ``(solution_set, condition)`` describing one parameter case.
    """
    a, b = sympify(a), sympify(b)
    if a.has(x) or b.has(x):
        raise ValueError("Coefficients must not contain the solve variable")

    branches = []
    nonzero = Ne(a, 0)
    if nonzero is not S.false:
        branches.append((FiniteSet(-b/a), nonzero))

    zero = And(Eq(a, 0), Eq(b, 0))
    if zero is not S.false:
        branches.append((S.Complexes, zero))

    inconsistent = And(Eq(a, 0), Ne(b, 0))
    if inconsistent is not S.false:
        branches.append((S.EmptySet, inconsistent))

    return tuple(branches)


def _linear_polynomial_solution_branches(expr, x):
    """Return guarded solution sets for a polynomial of degree at most one."""
    p = Poly(expr, x)
    if p.degree() > 1:
        raise ValueError("Expected a polynomial of degree at most one")
    return _linear_solution_branches(p.coeff_monomial(x), p.coeff_monomial(1), x)


def _quadratic_solution_branches(a, b, c, x):
    """Return guarded solution sets for ``a*x**2 + b*x + c = 0``."""
    a, b, c = sympify(a), sympify(b), sympify(c)
    if a.has(x) or b.has(x) or c.has(x):
        raise ValueError("Coefficients must not contain the solve variable")

    branches = []
    nonzero_a = Ne(a, 0)
    if nonzero_a is not S.false:
        discriminant = b**2 - 4*a*c
        root = sqrt(discriminant)
        solutions = FiniteSet((-b + root)/(2*a), (-b - root)/(2*a))
        branches.append((solutions, nonzero_a))

    zero_a = Eq(a, 0)
    if zero_a is not S.false:
        for solution, condition in _linear_solution_branches(b, c, x):
            condition = And(zero_a, condition)
            if condition is not S.false:
                branches.append((solution, condition))

    return tuple(branches)


def _quadratic_polynomial_solution_branches(expr, x):
    """Return guarded solution sets for a polynomial of degree at most two."""
    p = Poly(expr, x)
    if p.degree() > 2:
        raise ValueError("Expected a polynomial of degree at most two")
    return _quadratic_solution_branches(
        p.coeff_monomial(x**2), p.coeff_monomial(x), p.coeff_monomial(1), x)


def _conditional_union(branches, x):
    """Encode guarded branches as a union of ConditionSets."""
    return Union(*(ConditionSet(x, condition, solution)
                   for solution, condition in branches))


def _specialize_branches(branches, substitutions):
    """Specialize guarded branches, checking each guard before its set."""
    result = []
    for solution, condition in branches:
        condition = condition.xreplace(substitutions)
        if condition is S.false:
            continue
        solution = solution.xreplace(substitutions)
        if condition is S.true:
            result.append((solution, S.true))
        else:
            result.append((solution, condition))
    return tuple(result)


def _specialize_conditional_union(branches, x, substitutions):
    """Specialize guarded branches and return their ConditionSet union."""
    active = _specialize_branches(branches, substitutions)
    return _conditional_union(active, x)
