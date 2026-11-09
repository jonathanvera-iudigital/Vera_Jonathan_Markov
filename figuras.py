"""Figuras (300 dpi) a partir de los CSV de resultados/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experimentos import CARAS, ESCENARIOS, ESTADOS, modelo_de, rng_de

COL = ["#4C78A8", "#E45756", "#54A24B"]  # Justo, Cargado-6, Cargado-bajo
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 100})


def _guardar(fig, out, nombre):
    fig.tight_layout()
    fig.savefig(Path(out) / nombre, dpi=300)
    plt.close(fig)


def fig_modelo(out):
    m = modelo_de(ESCENARIOS[0])
    fig = plt.figure(figsize=(9, 5.4))
    ax = fig.add_axes([0, 0.36, 1, 0.64]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0.52, 1.0); ax.set_aspect("equal", adjustable="box")
    pos = [(0.17, 0.78), (0.5, 0.78), (0.83, 0.78)]
    r_nodo = 0.07
    for i, (x, y) in enumerate(pos):
        ax.add_patch(plt.Circle((x, y), r_nodo, color=COL[i], alpha=.9, zorder=3))
        ax.text(x, y, ESTADOS[i].replace("-", chr(10)), ha="center", va="center", color="white", fontsize=8.5,
                fontweight="bold", zorder=4)
        ax.text(x, y + 0.095, f"$\pi_{i}$={m.pi[i]:.1f}   $a_{{{i}{i}}}$={m.A[i, i]:.2f}", ha="center", fontsize=9)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            (x0, y0), (x1, y1) = pos[i], pos[j]
            rad = 0.5 if abs(i - j) == 2 else 0.3
            sgn = np.sign(x1 - x0)
            ax.annotate("", xy=(x1 - sgn * r_nodo, y1), xytext=(x0 + sgn * r_nodo, y0), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1, shrinkA=0, shrinkB=0,
                                        connectionstyle=f"arc3,rad={rad}"))
    ax.text(0.5, 0.545, "Las flechas indican transiciones posibles; sus probabilidades estan en la matriz $A$ (panel derecho; J=Justo, C6=Cargado-6, Cb=Cargado-bajo).",
            ha="center", fontsize=8, color="0.3")
    for i in range(3):
        a = fig.add_axes([0.07 + 0.235 * i, 0.08, 0.17, 0.22])
        a.bar(CARAS, m.B[i], color=COL[i]); a.set_ylim(0, 0.6); a.set_xticks(CARAS)
        a.set_title(f"$B$: {ESTADOS[i]}", fontsize=8); a.set_xlabel("cara observada", fontsize=8)
        a.tick_params(labelsize=7)
        if i == 0:
            a.set_ylabel("P(cara | estado)", fontsize=8)
    a = fig.add_axes([0.79, 0.08, 0.2, 0.26])
    a.imshow(m.A, cmap="Blues", vmin=0, vmax=1)
    for r in range(3):
        for c in range(3):
            a.text(c, r, f"{m.A[r, c]:.2f}", ha="center", va="center", fontsize=8,
                   color="white" if m.A[r, c] > .5 else "black")
    a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(["J", "C6", "Cb"], fontsize=7)
    a.set_yticklabels(["J", "C6", "Cb"], fontsize=7)
    a.set_xlabel("estado en $t+1$", fontsize=8); a.set_ylabel("estado en $t$", fontsize=8)
    a.set_title("Matriz $A$", fontsize=8)
    for s in a.spines.values():
        s.set_visible(True)
    fig.savefig(Path(out) / "fig1_modelo.png", dpi=300)
    plt.close(fig)


def fig_ejemplo(seed, out):
    m = modelo_de(ESCENARIOS[0])
    q, o = m.sample(400, rng_de(seed, 9))
    v = m.viterbi(o)[0]
    g = m.posterior(o)
    fig, ax = plt.subplots(4, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.6]})
    t = np.arange(400)
    for a, x, tit in [(ax[0], q, "Estado oculto real"), (ax[1], v, "Estado decodificado (Viterbi)")]:
        a.scatter(t, x, c=[COL[k] for k in x], s=6, marker="s")
        a.set_yticks(range(3)); a.set_yticklabels(ESTADOS, fontsize=8); a.set_title(tit, fontsize=9, loc="left")
    ax[2].scatter(t, o + 1, c="0.3", s=5); ax[2].set_ylabel("cara"); ax[2].set_title("Observaciones (tiradas)", fontsize=9, loc="left")
    for i in range(3):
        ax[3].plot(t, g[:, i], color=COL[i], lw=1, label=ESTADOS[i])
    ax[3].set_ylabel("P(estado | datos)"); ax[3].set_xlabel("tirada $t$"); ax[3].legend(ncol=3, fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax[3].set_title("Probabilidad posterior $\\gamma_t(i)$", fontsize=9, loc="left")
    _guardar(fig, out, "fig2_ejemplo_secuencia.png")


def fig_escenarios(out):
    df = pd.read_csv(Path(out) / "crudo_escenarios.csv")
    esc = list(df.escenario.unique())
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    series = [("acc_viterbi", "HMM (parametros reales) - Viterbi", COL[0]),
              ("acc_viterbi_aprendido", "Baum-Welch (init. aleatoria)", "#F2B134"),
              ("acc_viterbi_aprendido_peg", "Baum-Welch (init. pegajosa)", COL[2]),
              ("acc_ingenuo", "Ingenuo: siempre 'Justo'", "0.6")]
    w = 0.2
    for s, (c, lab, col) in enumerate(series):
        bp = ax[0].boxplot([df[df.escenario == e][c] for e in esc], positions=np.arange(len(esc)) + (s - 1.5) * w,
                           widths=w * .9, patch_artist=True, showfliers=True,
                           flierprops=dict(markersize=2.5))
        for b in bp["boxes"]: b.set_facecolor(col); b.set_alpha(.8)
        for md in bp["medians"]: md.set_color("black")
        ax[0].plot([], [], color=col, lw=6, label=lab)
    ax[0].set_xticks(range(len(esc))); ax[0].set_xticklabels([e.replace(" ", "\n", 1) for e in esc], fontsize=8)
    ax[0].set_ylabel("exactitud por tirada"); ax[0].set_title("(a) Exactitud de decodificacion de 3 estados", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    for s, (c, lab, col) in enumerate([("f1_viterbi", "HMM - Viterbi", COL[0]), ("f1_ventana", "Ventana chi-cuadrado", "0.6")]):
        bp = ax[1].boxplot([df[df.escenario == e][c] for e in esc], positions=np.arange(len(esc)) + (s - .5) * .35,
                           widths=.3, patch_artist=True, flierprops=dict(markersize=2.5))
        for b in bp["boxes"]: b.set_facecolor(col); b.set_alpha(.8)
        for md in bp["medians"]: md.set_color("black")
        ax[1].plot([], [], color=col, lw=6, label=lab)
    ax[1].set_xticks(range(len(esc))); ax[1].set_xticklabels([e.replace(" ", "\n", 1) for e in esc], fontsize=8)
    ax[1].set_ylabel("$F_1$ (detectar dado cargado)"); ax[1].set_title("(b) Deteccion de trampa: HMM vs linea base", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _guardar(fig, out, "fig3_escenarios.png")


def fig_inicializacion(out):
    h = pd.read_pickle(Path(out) / "_hist_bw.pkl")
    d = pd.read_csv(Path(out) / "sens_inicializacion.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    n = 2000
    cols = {0.0: "#F2B134", 0.8: COL[2]}
    for x, st in zip(h, d.sticky):
        ax[0].plot(np.arange(1, len(x) + 1), np.array(x) / n, lw=.8, alpha=.6, color=cols[st])
    ax[0].plot([], [], color=cols[0.0], label="aleatoria"); ax[0].plot([], [], color=cols[0.8], label="pegajosa (0.8)")
    ax[0].legend(frameon=False, fontsize=8, title="inicializacion de $A$", title_fontsize=8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("iteracion de Baum-Welch (log)"); ax[0].set_ylabel("log-verosimilitud por tirada")
    top = max(x[-1] for x in h) / n
    ax[0].set_ylim(top - 0.12, top + 0.01)
    ax[0].set_title("(a) Convergencia desde 30 inicializaciones por estrategia", fontsize=9)
    for st, c in cols.items():
        e = d[d.sticky == st]
        ax[1].scatter(e.ll_final / n, e.err_B, c=c, s=18, alpha=.8)
    ax[1].set_xlabel("log-verosimilitud final por tirada"); ax[1].set_ylabel("error absoluto medio de $B$")
    ax[1].set_title("(b) Optimos locales vs error del parametro", fontsize=9)
    _guardar(fig, out, "fig4_inicializacion.png")


def fig_sensibilidad(out):
    p = pd.read_csv(Path(out) / "sens_perturbacion.csv")
    p = p[p.metrica == "acc_viterbi"]
    l = pd.read_csv(Path(out) / "sens_longitud.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    x = p["eps"].astype(float)
    ax[0].errorbar(x, p["media"], yerr=[p["media"] - p["ic95_inf"], p["ic95_sup"] - p["media"]], marker="o",
                   color=COL[1], capsize=3)
    ax[0].set_xlabel("nivel de perturbacion $\\varepsilon$ en $A$ y $B$"); ax[0].set_ylabel("exactitud Viterbi")
    ax[0].set_title("(a) Sensibilidad a errores de especificacion (IC 95%)", fontsize=9)
    T = l["T"].astype(float)
    for met, lab, col in [("acc_aprendido", "Baum-Welch (aleatoria)", "#F2B134"), ("acc_aprendido_peg", "Baum-Welch (pegajosa)", COL[2]), ("acc_oraculo", "parametros reales", COL[0])]:
        d = l[l.metrica == met]
        ax[1].errorbar(d["T"].astype(float), d["media"], yerr=[d["media"] - d["ic95_inf"], d["ic95_sup"] - d["media"]],
                       marker="o", capsize=3, label=lab, color=col)
    ax[1].set_xscale("log"); ax[1].set_xlabel("longitud de la secuencia $T$"); ax[1].set_ylabel("exactitud Viterbi")
    ax[1].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3); ax[1].set_title("(b) Efecto de la cantidad de datos (IC 95%)", fontsize=9)
    _guardar(fig, out, "fig5_sensibilidad.png")


def hacer_figuras(seed, out):
    out = Path(out)
    fig_modelo(out); fig_ejemplo(seed, out)
    for f in (fig_escenarios, fig_inicializacion, fig_sensibilidad):
        try:
            f(out)
        except FileNotFoundError as e:
            print("omitida", f.__name__, e)
