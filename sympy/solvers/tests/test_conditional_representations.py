from __future__ import annotations

import pytest

from sympy import (ConditionSet, Contains, Eq, FiniteSet, Ne, S, Union,
                   nan, solveset, symbols, zoo)
from sympy.logic.boolalg import ITE
from sympy.solvers.conditional import (
    _conditional_union, _linear_solution_branches)


@pytest.fixture
def encodings():
    a, b, x = symbols('a b x', finite=True)
    branches = _linear_solution_branches(a, b, x)
    return a, b, x, (
        ('guarded ConditionSet union', _conditional_union(branches, x)),
        ('equation ConditionSet',
         ConditionSet(x, Eq(a*x + b, 0), S.Complexes)),
        ('ITE membership ConditionSet', ConditionSet(
            x, ITE(Ne(a, 0), Contains(x, FiniteSet(-b/a)), Eq(b, 0)),
            S.Complexes)),
    )


@pytest.mark.parametrize('name', [
    'guarded ConditionSet union',
    'equation ConditionSet',
    'ITE membership ConditionSet',
])
def test_conditional_representation_full_specialization(encodings, name):
    a, b, x, representations = encodings
    representation = dict(representations)[name]
    for substitutions, expected in (
        ({a: 2, b: 4}, FiniteSet(-2)),
        ({a: 0, b: 0}, S.Complexes),
        ({a: 0, b: 1}, S.EmptySet),
    ):
        actual = representation.xreplace(substitutions)
        reference = solveset((a*x + b).xreplace(substitutions), x, S.Complexes)
        if name == 'equation ConditionSet' and substitutions == {a: 2, b: 4}:
            assert actual == ConditionSet(x, Eq(2*x + 4, 0), S.Complexes)
            assert actual.contains(-2) is S.true
            assert actual.contains(0) is S.false
        else:
            assert actual == expected == reference


@pytest.mark.parametrize('name', [
    'guarded ConditionSet union',
    'equation ConditionSet',
    'ITE membership ConditionSet',
])
def test_conditional_representation_partial_specialization(encodings, name):
    a, b, x, representations = encodings
    representation = dict(representations)[name]
    expected = ConditionSet(x, Eq(b, 0), S.Complexes)
    assert representation.xreplace({a: 0}) == expected


@pytest.mark.parametrize('name', [
    'guarded ConditionSet union',
    'equation ConditionSet',
    'ITE membership ConditionSet',
])
def test_conditional_representation_sequential_replacements(encodings, name):
    a, b, _, representations = encodings
    representation = dict(representations)[name]
    ab = representation.xreplace({a: 0}).xreplace({b: 1})
    ba = representation.xreplace({b: 1}).xreplace({a: 0})
    assert ab == ba == S.EmptySet


@pytest.mark.parametrize('name', [
    'guarded ConditionSet union',
    'equation ConditionSet',
    'ITE membership ConditionSet',
])
def test_conditional_representation_membership(encodings, name):
    a, b, _, representations = encodings
    representation = dict(representations)[name]
    for value, cases in (
        (0, (({a: 2, b: 4}, False), ({a: 0, b: 0}, True),
             ({a: 0, b: 1}, False), ({a: 2, b: 0}, True))),
        (1, (({a: 2, b: 4}, False), ({a: 0, b: 0}, True),
             ({a: 0, b: 1}, False), ({a: 1, b: -1}, True))),
    ):
        membership = representation.contains(value)
        assert membership.free_symbols <= {a, b}
        for substitutions, expected in cases:
            actual = membership.xreplace(substitutions)
            assert actual is (S.true if expected else S.false)

    if name == 'equation ConditionSet':
        assert representation.contains(0) == Eq(b, 0)
        assert representation.contains(1) == Eq(a + b, 0)
    elif name == 'ITE membership ConditionSet':
        assert representation.contains(0) == ITE(
            Ne(a, 0), Contains(0, FiniteSet(-b/a)), Eq(b, 0))
        assert representation.contains(1) == ITE(
            Ne(a, 0), Contains(1, FiniteSet(-b/a)), Eq(b, 0))
    else:
        assert representation.contains(0) == (
            (Eq(a, 0) & Eq(b, 0)) |
            (Ne(a, 0) & Contains(0, FiniteSet(-b/a))))
        assert representation.contains(1) == (
            (Eq(a, 0) & Eq(b, 0)) |
            (Ne(a, 0) & Contains(1, FiniteSet(-b/a))))


def test_conditional_representation_branch_extraction(encodings):
    a, b, _, representations = encodings
    guarded = dict(representations)['guarded ConditionSet union']
    extracted = {
        (branch.base_set, branch.condition)
        for branch in guarded.args if isinstance(branch, ConditionSet)
    }
    assert extracted == {
        (FiniteSet(-b/a), Ne(a, 0)),
        (S.Complexes, Eq(a, 0) & Eq(b, 0)),
    }
    assert isinstance(dict(representations)['equation ConditionSet'], ConditionSet)
    assert isinstance(dict(representations)['ITE membership ConditionSet'].condition,
                      ITE)


def test_conditional_representation_inactive_division_branches(encodings):
    a, b, x, representations = encodings
    assert _linear_solution_branches(0, 1, x) == ((S.EmptySet, S.true),)
    for _, representation in representations:
        result = representation.xreplace({a: 0, b: 1})
        assert result is S.EmptySet
        assert not result.has(zoo, nan)


def test_conditional_representation_symbol_binding(encodings):
    a, b, x, representations = encodings
    for _, representation in representations:
        assert representation.free_symbols == {a, b}
        condition_sets = (representation.args if isinstance(representation, Union)
                          else (representation,))
        assert all(x in condition_set.bound_symbols
                   for condition_set in condition_sets
                   if isinstance(condition_set, ConditionSet))
