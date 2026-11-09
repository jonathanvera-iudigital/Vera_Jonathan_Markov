import itertools

import numpy as np
import pytest

from hmm import HMM, align_states, permute
from experimentos import ESCENARIOS, modelo_de


def modelo_pequeno():
    return HMM([0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]], [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])


def test_matrices_invalidas():
    with pytest.raises(ValueError):
        HMM([0.5, 0.6], [[1, 0], [0, 1]], [[1, 0], [0, 1]])


def test_forward_igual_enumeracion():
    m, o = modelo_pequeno(), np.array([0, 2, 1, 2, 0])
    total = 0.0
    for q in itertools.product(range(2), repeat=len(o)):
        p = m.pi[q[0]] * m.B[q[0], o[0]]
        for t in range(1, len(o)):
            p *= m.A[q[t - 1], q[t]] * m.B[q[t], o[t]]
        total += p
    assert np.isclose(m.loglik(o), np.log(total))


def test_viterbi_igual_fuerza_bruta():
    m, o = modelo_pequeno(), np.array([0, 2, 1, 2, 0, 1])
    mejor, mp = None, -1
    for q in itertools.product(range(2), repeat=len(o)):
        p = m.pi[q[0]] * m.B[q[0], o[0]]
        for t in range(1, len(o)):
            p *= m.A[q[t - 1], q[t]] * m.B[q[t], o[t]]
        if p > mp:
            mejor, mp = q, p
    path, lp = m.viterbi(o)
    assert tuple(path) == mejor and np.isclose(lp, np.log(mp))


def test_posterior_suma_uno():
    g = modelo_pequeno().posterior(np.array([0, 2, 1, 2]))
    assert np.allclose(g.sum(1), 1)


def test_baum_welch_monotono():
    m = modelo_pequeno()
    rng = np.random.default_rng(1)
    _, o = m.sample(500, rng)
    _, hist = HMM.random_init(2, 3, rng).baum_welch(o, max_iter=40, tol=0)
    assert np.all(np.diff(hist) > -1e-8)


def test_muestreo_reproducible():
    m = modelo_de(ESCENARIOS[0])
    a = m.sample(300, np.random.default_rng(7))
    b = m.sample(300, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_frecuencias_emision():
    m = modelo_de(ESCENARIOS[0])
    q, o = m.sample(60000, np.random.default_rng(3))
    for i in range(m.N):
        f = np.bincount(o[q == i], minlength=6) / (q == i).sum()
        assert np.abs(f - m.B[i]).max() < 0.03


def test_align_states():
    m = modelo_de(ESCENARIOS[0])
    perm = np.array([2, 0, 1])
    est = permute(m, perm)
    assert np.array_equal(est.B[align_states(m, est)], m.B)


def test_contra_hmmlearn():
    hl = pytest.importorskip("hmmlearn.hmm")
    m = modelo_de(ESCENARIOS[0])
    q, o = m.sample(400, np.random.default_rng(5))
    ref = hl.CategoricalHMM(n_components=3)
    ref.startprob_, ref.transmat_, ref.emissionprob_ = m.pi, m.A, m.B
    X = o.reshape(-1, 1)
    assert np.isclose(ref.score(X), m.loglik(o))
    assert np.array_equal(ref.decode(X)[1], m.viterbi(o)[0])
