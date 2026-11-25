import numpy as np
import pytest
from src.analysis.cka_tracker import linear_cka, linear_hsic


def test_cka_identical():
    X = np.random.randn(50, 32)
    assert abs(linear_cka(X, X) - 1.0) < 1e-6


def test_cka_orthogonal_invariance():
    X = np.random.randn(100, 16)
    Q, _ = np.linalg.qr(np.random.randn(16, 16))
    Y = X @ Q
    assert abs(linear_cka(X, Y) - 1.0) < 1e-4


def test_cka_range():
    X = np.random.randn(80, 32)
    Y = np.random.randn(80, 32)
    score = linear_cka(X, Y)
    assert 0 <= score <= 1


def test_cka_symmetry():
    X = np.random.randn(60, 20)
    Y = np.random.randn(60, 20)
    assert abs(linear_cka(X, Y) - linear_cka(Y, X)) < 1e-10


def test_hsic_positive():
    X = np.random.randn(50, 10)
    assert linear_hsic(X, X) >= 0
