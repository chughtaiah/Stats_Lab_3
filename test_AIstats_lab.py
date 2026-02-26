import AI_stats_lab as A
import math


def test_card_experiment():
    P_A, P_B, P_B_given_A, P_AB, emp_A, emp_B_given_A, error = A.card_experiment()

    assert abs(P_A - 4/52) < 1e-12
    assert abs(P_B_given_A - 3/51) < 1e-12
    assert abs(emp_A - P_A) < 0.01
    assert abs(emp_B_given_A - P_B_given_A) < 0.02


def test_bernoulli_lightbulb():
    theoretical, theoretical_zero, empirical, error = A.bernoulli_lightbulb()

    assert abs(theoretical - 0.05) < 1e-12
    assert abs(theoretical_zero - 0.95) < 1e-12
    assert abs(empirical - theoretical) < 0.01


def test_binomial_bulbs():
    P0, P2, P_ge_1, emp_ge_1, error = A.binomial_bulbs()

    expected_P0 = (0.95)**10
    assert abs(P0 - expected_P0) < 1e-12
    assert abs(emp_ge_1 - P_ge_1) < 0.02


def test_geometric_die():
    P1, P3, P_gt_4, emp_gt_4, error = A.geometric_die()

    assert abs(P1 - (1/6)) < 1e-12
    assert abs(P3 - ((5/6)**2)*(1/6)) < 1e-12
    assert abs(emp_gt_4 - P_gt_4) < 0.02


def test_poisson_customers():
    P0, P15, P_ge_18, emp_ge_18, error = A.poisson_customers()

    expected_P0 = math.exp(-12)
    assert abs(P0 - expected_P0) < 1e-12
    assert abs(emp_ge_18 - P_ge_18) < 0.02
