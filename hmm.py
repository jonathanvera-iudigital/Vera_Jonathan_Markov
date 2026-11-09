"""Modelo Oculto de Markov (HMM) discreto implementado desde cero con NumPy.

Notacion (Rabiner, 1989):
    S = {s_0..s_{N-1}}  estados ocultos        V = {v_0..v_{M-1}}  simbolos observables
    pi[i]   = P(q_1 = s_i)                      distribucion inicial
    A[i,j]  = P(q_{t+1} = s_j | q_t = s_i)      matriz de transicion
    B[i,k]  = P(o_t = v_k | q_t = s_i)          matriz de emision

Algoritmos: muestreo, Forward/Backward con escalado, Viterbi (log), decodificacion
posterior y Baum-Welch (EM) con varias secuencias.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _check_stochastic(M: np.ndarray, name: str) -> None:
    if np.any(M < 0) or not np.allclose(M.sum(axis=-1), 1.0, atol=1e-8):
        raise ValueError(f"{name} debe ser estocastica (filas >= 0 que suman 1)")


class HMM:
    def __init__(self, pi, A, B):
        self.pi = np.asarray(pi, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        n, m = self.B.shape
        if self.pi.shape != (n,) or self.A.shape != (n, n):
            raise ValueError("dimensiones incompatibles entre pi, A y B")
        _check_stochastic(self.pi, "pi")
        _check_stochastic(self.A, "A")
        _check_stochastic(self.B, "B")

    @property
    def N(self) -> int:
        return self.B.shape[0]

    @property
    def M(self) -> int:
        return self.B.shape[1]

    # ------------------------------------------------------------ muestreo
    def sample(self, T: int, rng: np.random.Generator):
        """Genera (estados, observaciones) de longitud T por transformada inversa."""
        cA, cB, cpi = self.A.cumsum(1), self.B.cumsum(1), self.pi.cumsum()
        u = rng.random((T, 2))
        q = np.empty(T, dtype=int)
        o = np.empty(T, dtype=int)
        s = min(int(np.searchsorted(cpi, u[0, 0], side="right")), self.N - 1)
        for t in range(T):
            q[t] = s
            o[t] = min(int(np.searchsorted(cB[s], u[t, 1], side="right")), self.M - 1)
            if t + 1 < T:
                s = min(int(np.searchsorted(cA[s], rng.random(), side="right")), self.N - 1)
        return q, o

    # ------------------------------------------------------ forward/backward
    def forward(self, obs):
        """Forward escalado. Devuelve (alpha_hat, c, log P(O|lambda))."""
        T = len(obs)
        alpha = np.empty((T, self.N))
        c = np.empty(T)
        a = self.pi * self.B[:, obs[0]]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            c[t] = a.sum()
            if c[t] <= 0:
                raise ValueError("observacion con probabilidad 0 bajo el modelo")
            alpha[t] = a / c[t]
        return alpha, c, float(np.log(c).sum())

    def backward(self, obs, c):
        T = len(obs)
        beta = np.empty((T, self.N))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]
        return beta

    def loglik(self, obs) -> float:
        return self.forward(obs)[2]

    def posterior(self, obs):
        """gamma[t,i] = P(q_t = s_i | O, lambda)."""
        alpha, c, _ = self.forward(obs)
        beta = self.backward(obs, c)
        g = alpha * beta
        return g / g.sum(axis=1, keepdims=True)

    # -------------------------------------------------------------- Viterbi
    def viterbi(self, obs):
        """Secuencia de estados mas probable (en log). Devuelve (camino, log-prob)."""
        T = len(obs)
        with np.errstate(divide="ignore"):
            lpi, lA, lB = np.log(self.pi), np.log(self.A), np.log(self.B)
        delta = np.empty((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = lpi + lB[:, obs[0]]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + lA
            psi[t] = cand.argmax(0)
            delta[t] = cand.max(0) + lB[:, obs[t]]
        path = np.empty(T, dtype=int)
        path[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, float(delta[-1].max())

    # ----------------------------------------------------------- Baum-Welch
    def baum_welch(self, seqs, max_iter=200, tol=1e-5, smooth=1e-6):
        """EM sobre una lista de secuencias. Devuelve (modelo_nuevo, historial_loglik).

        tol se aplica a la mejora de log-verosimilitud por observacion.
        """
        if isinstance(seqs, np.ndarray) and seqs.ndim == 1:
            seqs = [seqs]
        n_obs = sum(len(s) for s in seqs)
        pi, A, B = self.pi.copy(), self.A.copy(), self.B.copy()
        hist = []
        for _ in range(max_iter):
            m = HMM(pi, A, B)
            pi_n = np.zeros(m.N)
            A_n = np.zeros((m.N, m.N))
            B_n = np.zeros((m.N, m.M))
            ll = 0.0
            for o in seqs:
                alpha, c, l = m.forward(o)
                beta = m.backward(o, c)
                g = alpha * beta
                g /= g.sum(1, keepdims=True)
                w = m.B[:, o[1:]].T * beta[1:] / c[1:, None]  # (T-1, N)
                A_n += m.A * (alpha[:-1].T @ w)
                pi_n += g[0]
                for k in range(m.M):
                    B_n[:, k] += g[o == k].sum(0)
                ll += l
            hist.append(ll)
            pi = (pi_n + smooth) / (pi_n + smooth).sum()
            A = (A_n + smooth) / (A_n + smooth).sum(1, keepdims=True)
            B = (B_n + smooth) / (B_n + smooth).sum(1, keepdims=True)
            if len(hist) > 1 and (hist[-1] - hist[-2]) / n_obs < tol:
                break
        return HMM(pi, A, B), hist

    @staticmethod
    def random_init(N, M, rng, sticky=0.0):
        """Inicializacion aleatoria (Dirichlet) para Baum-Welch.

        sticky in [0,1): mezcla la fila de A con la identidad (regimenes persistentes), practica comun
        cuando se espera que los estados duren varias observaciones. sticky=0 -> aleatoria pura.
        """
        A = rng.dirichlet(np.ones(N) * 5, size=N)
        A = sticky * np.eye(N) + (1 - sticky) * A
        return HMM(rng.dirichlet(np.ones(N)), A, rng.dirichlet(np.ones(M) * 2, size=N))


def align_states(true: HMM, est: HMM) -> np.ndarray:
    """Permutacion perm tal que est.B[perm] se parece mas a true.B (resuelve label switching)."""
    from itertools import permutations
    best, best_d = None, np.inf
    for p in permutations(range(est.N)):
        d = np.abs(est.B[list(p)] - true.B).sum()
        if d < best_d:
            best, best_d = np.array(p), d
    return best


def permute(model: HMM, perm) -> HMM:
    perm = np.asarray(perm)
    return HMM(model.pi[perm], model.A[np.ix_(perm, perm)], model.B[perm])
