from __future__ import annotations

from sympy import ConditionSet, Eq, S, symbols


def test_conditionset_subs_parameter_guard_experiment():
    a, x = symbols('a x')
    condition_set = ConditionSet(x, Eq(a, 0), S.Complexes)
    assert condition_set.subs(a, 2) == condition_set
    assert condition_set.xreplace({a: 2}) == S.EmptySet
