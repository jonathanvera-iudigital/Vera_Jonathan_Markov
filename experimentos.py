"""Experimentos del HMM de deteccion de dados cargados.

Uso:
    python experimentos.py demo                 # una secuencia narrada
    python experimentos.py todo [--rapido]      # todos los experimentos + figuras
    python experimentos.py escenarios|inicializacion|perturbacion|longitud|figuras|reproducibilidad
Opciones: --seed 42  --salida resultados  --rapido (menos replicas)  --workers N
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hmm import HMM, align_states, permute

ESTADOS = ["Justo", "Cargado-6", "Cargado-bajo"]
CARAS = [1, 2, 3, 4, 5, 6]
B_JUSTO = [1 / 6] * 6


def _esc(nombre, desc, b6, bbajo, A):
    return dict(nombre=nombre, desc=desc, pi=[0.8, 0.1, 0.1],
                A=A, B=[B_JUSTO, b6, bbajo], T=2000)


A_BASE = [[0.98, 0.01, 0.01], [0.03, 0.95, 0.02], [0.03, 0.02, 0.95]]
A_RAPIDA = [[0.95, 0.025, 0.025], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]]
ESCENARIOS = [
    _esc("E1 Base", "sesgo moderado, regimenes persistentes",
         [.1, .1, .1, .1, .1, .5], [.3, .3, .1, .1, .1, .1], A_BASE),
    _esc("E2 Sesgo debil", "sesgo leve (P(6)=0.30)",
         [.14, .14, .14, .14, .14, .30], [.22, .22, .14, .14, .14, .14], A_BASE),
    _esc("E3 Cambios frecuentes", "regimenes cortos (permanencia 0.90-0.95)",
         [.1, .1, .1, .1, .1, .5], [.3, .3, .1, .1, .1, .1], A_RAPIDA),
    _esc("E4 Sesgo fuerte", "sesgo extremo (P(6)=0.80)",
         [.04, .04, .04, .04, .04, .80], [.45, .45, .025, .025, .025, .025], A_BASE),
]


def modelo_de(esc) -> HMM:
    return HMM(esc["pi"], esc["A"], esc["B"])


def rng_de(seed, *claves) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *claves]))


# ----------------------------------------------------------------- metricas
def f1_cargado(q_real, q_est) -> float:
    r, e = q_real != 0, q_est != 0
    tp = (r & e).sum()
    d = 2 * tp + (~r & e).sum() + (r & ~e).sum()
    return float(2 * tp / d) if d else 1.0


def detector_ventana(obs, w=31, umbral=11.07) -> np.ndarray:
    """Linea base sin HMM: chi-cuadrado de uniformidad en ventana centrada.
    Marca 'cargado' (1) si chi2 > umbral (chi2_{0.95,5}=11.07)."""
    T = len(obs)
    cum = np.zeros((T + 1, 6))
    cum[1:] = np.cumsum(np.eye(6)[obs], axis=0)
    out = np.zeros(T, dtype=int)
    for t in range(T):
        a, b = max(0, t - w // 2), min(T, t + w // 2 + 1)
        n = b - a
        cnt = cum[b] - cum[a]
        out[t] = ((cnt - n / 6) ** 2 / (n / 6)).sum() > umbral
    return out


STICKY = 0.8  # inicializacion "pegajosa" de A (ver hmm.HMM.random_init)


def ajustar_bw(obs, rng, restarts=3, max_iter=300, sticky=0.0):
    mejor = None
    for _ in range(restarts):
        m, h = HMM.random_init(3, 6, rng, sticky).baum_welch(obs, max_iter=max_iter)
        if mejor is None or h[-1] > mejor[1][-1]:
            mejor = (m, h)
    return mejor


def _replica_escenario(args):
    seed, k, r, restarts = args
    esc = ESCENARIOS[k]
    real = modelo_de(esc)
    rng = rng_de(seed, 1, k, r)
    q, o = real.sample(esc["T"], rng)
    vit, _ = real.viterbi(o)
    post = real.posterior(o).argmax(1)
    base = detector_ventana(o)
    fila = dict(escenario=esc["nombre"], rep=r,
                acc_viterbi=(vit == q).mean(), acc_posterior=(post == q).mean(),
                f1_viterbi=f1_cargado(q, vit), f1_ventana=f1_cargado(q, base),
                acc_ventana=((base != 0) == (q != 0)).mean(),
                acc_ingenuo=(q == 0).mean(),  # clasificador que siempre dice 'Justo'
                ll_oraculo=real.loglik(o) / len(o))
    for suf, st in (("", 0.0), ("_peg", STICKY)):
        est, hist = ajustar_bw(o, rng, restarts, sticky=st)
        est = permute(est, align_states(real, est))
        pv, _ = est.viterbi(o)
        fila.update({f"acc_viterbi_aprendido{suf}": (pv == q).mean(), f"f1_aprendido{suf}": f1_cargado(q, pv),
                     f"iter_bw{suf}": len(hist), f"err_A{suf}": np.abs(est.A - real.A).mean(),
                     f"err_B{suf}": np.abs(est.B - real.B).mean()})
    return fila


def _replica_init(args):
    seed, j, o, real, sticky = args
    rng = rng_de(seed, 2, j)
    m, h = HMM.random_init(3, 6, rng, sticky).baum_welch(o, max_iter=300)
    m = permute(m, align_states(real, m))
    return dict(init=j, sticky=sticky, ll_final=h[-1], iteraciones=len(h), err_B=np.abs(m.B - real.B).mean(),
                err_A=np.abs(m.A - real.A).mean(), hist=h)


def perturbar(m: HMM, eps, rng) -> HMM:
    """Ruido multiplicativo log-normal de nivel eps sobre A y B (renormalizado)."""
    def f(M):
        X = M * np.exp(eps * rng.standard_normal(M.shape))
        return X / X.sum(1, keepdims=True)
    return HMM(m.pi, f(m.A), f(m.B))


def _replica_perturb(args):
    seed, eps, r = args
    real = modelo_de(ESCENARIOS[0])
    rng = rng_de(seed, 3, int(eps * 1000), r)
    q, o = real.sample(2000, rng)
    pm = perturbar(real, eps, rng)
    return dict(eps=eps, rep=r, acc_viterbi=(pm.viterbi(o)[0] == q).mean(),
                dif_A=np.abs(pm.A - real.A).mean(), dif_B=np.abs(pm.B - real.B).mean())


def _replica_longitud(args):
    seed, T, r = args
    real = modelo_de(ESCENARIOS[0])
    rng = rng_de(seed, 4, T, r)
    q, o = real.sample(T, rng)
    fila = dict(T=T, rep=r, acc_oraculo=(real.viterbi(o)[0] == q).mean())
    for suf, st in (("", 0.0), ("_peg", STICKY)):
        est, hist = ajustar_bw(o, rng, restarts=3, sticky=st)
        est = permute(est, align_states(real, est))
        fila.update({f"acc_aprendido{suf}": (est.viterbi(o)[0] == q).mean(),
                     f"err_A{suf}": np.abs(est.A - real.A).mean(), f"err_B{suf}": np.abs(est.B - real.B).mean()})
    return fila


# ------------------------------------------------------------- estadistica
def resumen(df, cols, por):
    filas = []
    for g, d in df.groupby(por, sort=False):
        for c in cols:
            x = d[c].to_numpy(float)
            n = len(x)
            sd = x.std(ddof=1) if n > 1 else 0.0
            h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
            filas.append({por: g, "metrica": c, "n": n, "media": x.mean(), "sd": sd,
                          "min": x.min(), "max": x.max(),
                          "ic95_inf": x.mean() - h, "ic95_sup": x.mean() + h})
    return pd.DataFrame(filas)


def comparar(df, col, ref="E1 Base"):
    filas = []
    a = df[df.escenario == ref][col].to_numpy()
    for e in df.escenario.unique():
        if e == ref:
            continue
        b = df[df.escenario == e][col].to_numpy()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        u, pu = stats.mannwhitneyu(a, b)
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        filas.append(dict(metrica=col, referencia=ref, escenario=e, dif_medias=b.mean() - a.mean(),
                          t_welch=t, p_welch=p, p_mannwhitney=pu,
                          d_cohen=(b.mean() - a.mean()) / sp if sp else np.nan))
    return pd.DataFrame(filas)


def pool_map(fn, tareas, workers):
    if workers == 1:
        return [fn(t) for t in tareas]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, tareas, chunksize=1))


# ------------------------------------------------------------ experimentos
def exp_escenarios(seed, out, R, workers, restarts=3):
    tareas = [(seed, k, r, restarts) for k in range(len(ESCENARIOS)) for r in range(R)]
    df = pd.DataFrame(pool_map(_replica_escenario, tareas, workers))
    df.to_csv(out / "crudo_escenarios.csv", index=False)
    cols = ["acc_viterbi", "acc_posterior", "f1_viterbi", "f1_ventana", "acc_ventana",
            "acc_ingenuo", "acc_viterbi_aprendido", "f1_aprendido", "err_A", "err_B", "iter_bw",
            "acc_viterbi_aprendido_peg", "f1_aprendido_peg", "err_A_peg", "err_B_peg", "iter_bw_peg"]
    resumen(df, cols, "escenario").to_csv(out / "resumen_escenarios.csv", index=False)
    pd.concat([comparar(df, c) for c in ["acc_viterbi", "f1_viterbi", "acc_viterbi_aprendido_peg"]]
              ).to_csv(out / "comparacion_estadistica.csv", index=False)
    # HMM vs linea base (pareado, por escenario)
    filas = []
    for e, d in df.groupby("escenario", sort=False):
        w, p = stats.wilcoxon(d.f1_viterbi, d.f1_ventana)
        filas.append(dict(escenario=e, f1_hmm=d.f1_viterbi.mean(), f1_ventana=d.f1_ventana.mean(),
                          wilcoxon_p=p))
    pd.DataFrame(filas).to_csv(out / "hmm_vs_ventana.csv", index=False)
    return df


def exp_inicializacion(seed, out, n_init, workers):
    real = modelo_de(ESCENARIOS[0])
    _, o = real.sample(ESCENARIOS[0]["T"], rng_de(seed, 5))
    res = pool_map(_replica_init, [(seed, j, o, real, st) for st in (0.0, STICKY) for j in range(n_init)], workers)
    pd.DataFrame(res).drop(columns="hist").to_csv(out / "sens_inicializacion.csv", index=False)
    pd.to_pickle([r["hist"] for r in res], out / "_hist_bw.pkl")
    return res


def exp_perturbacion(seed, out, R, workers):
    epss = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    df = pd.DataFrame(pool_map(_replica_perturb, [(seed, e, r) for e in epss for r in range(R)], workers))
    df.to_csv(out / "sens_perturbacion_crudo.csv", index=False)
    resumen(df.assign(eps=df.eps.astype(str)), ["acc_viterbi", "dif_A", "dif_B"], "eps"
            ).to_csv(out / "sens_perturbacion.csv", index=False)
    return df


def exp_longitud(seed, out, R, workers, Ts=(100, 250, 500, 1000, 2000, 4000)):
    df = pd.DataFrame(pool_map(_replica_longitud, [(seed, T, r) for T in Ts for r in range(R)], workers))
    df.to_csv(out / "sens_longitud_crudo.csv", index=False)
    resumen(df.assign(T=df["T"].astype(str)), ["acc_aprendido", "acc_aprendido_peg", "acc_oraculo", "err_A", "err_B", "err_A_peg", "err_B_peg"], "T"
            ).to_csv(out / "sens_longitud.csv", index=False)
    return df


def escribir_entorno(out, seed):
    import hmmlearn, matplotlib, scipy
    txt = (f"Python {sys.version.split()[0]} ({platform.platform()})\nnumpy {np.__version__}\n"
           f"pandas {pd.__version__}\nscipy {scipy.__version__}\nmatplotlib {matplotlib.__version__}\n"
           f"hmmlearn {hmmlearn.__version__}\nsemilla_base {seed}\n")
    (out / "entorno.txt").write_text(txt, encoding="utf-8")


def reproducibilidad(seed, out):
    """Ejecuta dos veces un mini-experimento con la misma semilla y compara hashes SHA-256."""
    hs = []
    for _ in range(2):
        d = pd.DataFrame([_replica_escenario((seed, 0, r, 1)) for r in range(3)])
        hs.append(hashlib.sha256(d.round(12).to_csv(index=False).encode()).hexdigest())
    d3 = pd.DataFrame([_replica_escenario((seed + 1, 0, r, 1)) for r in range(3)])
    h3 = hashlib.sha256(d3.round(12).to_csv(index=False).encode()).hexdigest()
    txt = (f"corrida 1 (semilla {seed}):   {hs[0]}\ncorrida 2 (semilla {seed}):   {hs[1]}\n"
           f"corrida 3 (semilla {seed+1}): {h3}\nidentico con misma semilla: {hs[0] == hs[1]}\n"
           f"distinto con otra semilla:  {hs[0] != h3}\n")
    (out / "reproducibilidad.txt").write_text(txt, encoding="utf-8")
    print(txt)


def demo(seed):
    real = modelo_de(ESCENARIOS[0])
    q, o = real.sample(60, rng_de(seed, 0))
    vit, lp = real.viterbi(o)
    post = real.posterior(o)
    sim = {0: "J", 1: "6", 2: "b"}
    print("Tiradas      :", "".join(str(CARAS[x]) for x in o))
    print("Estado real  :", "".join(sim[x] for x in q))
    print("Viterbi      :", "".join(sim[x] for x in vit))
    print("Posterior max:", "".join(sim[x] for x in post.argmax(1)))
    print("(J=Justo, 6=Cargado-6, b=Cargado-bajo)")
    print(f"Exactitud Viterbi: {(vit == q).mean():.3f}   log P(O|lambda) = {real.loglik(o):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["demo", "todo", "escenarios", "inicializacion", "perturbacion",
                                    "longitud", "figuras", "reproducibilidad"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.salida)
    out.mkdir(exist_ok=True)
    R, Ri, Rp, Rl = (30, 30, 30, 20) if not a.rapido else (4, 6, 5, 3)
    if a.cmd == "demo":
        demo(a.seed)
        sys.exit()
    escribir_entorno(out, a.seed)
    if a.cmd in ("todo", "escenarios"):
        exp_escenarios(a.seed, out, R, a.workers)
    if a.cmd in ("todo", "inicializacion"):
        exp_inicializacion(a.seed, out, Ri, a.workers)
    if a.cmd in ("todo", "perturbacion"):
        exp_perturbacion(a.seed, out, Rp, a.workers)
    if a.cmd in ("todo", "longitud"):
        exp_longitud(a.seed, out, Rl, a.workers)
    if a.cmd in ("todo", "reproducibilidad"):
        reproducibilidad(a.seed, out)
    if a.cmd in ("todo", "figuras"):
        from figuras import hacer_figuras
        hacer_figuras(a.seed, out)
    print("Listo. Resultados en", out.resolve())
