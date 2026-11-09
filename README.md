# Detección de dados cargados con un Modelo Oculto de Markov (HMM)

Actividad didáctica 2 – Módulo 2. Un tramposo alterna entre un dado **justo**, uno **cargado al 6** y uno **cargado a
caras bajas**; solo vemos la cara que sale. Se modela con un HMM discreto (N = 3 estados ocultos, M = 6 símbolos), implementado
**desde cero con NumPy**, y se simulan 4 escenarios con réplicas, intervalos de confianza y análisis de sensibilidad.

## Modelo

- Estados ocultos S = {Justo, Cargado-6, Cargado-bajo}; observaciones V = {1..6}.
- π = [0,8; 0,1; 0,1]; A con diagonal dominante (0,98 / 0,95 / 0,95); B categórica por estado (dado justo = 1/6).
- Algoritmos en [`hmm.py`](hmm.py): muestreo, Forward/Backward escalados, Viterbi (log), decodificación posterior, Baum-Welch
  (EM, varias secuencias) y alineación de estados (label switching).

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python -m pytest -q                       # 9 pruebas (incluye validación contra hmmlearn)
python experimentos.py demo               # una secuencia narrada
python experimentos.py todo --seed 42     # ~8 min con 6 procesos; usa --rapido para probar y --workers N
```

Comandos: `demo`, `todo`, `escenarios`, `inicializacion`, `perturbacion`, `longitud`, `figuras`, `reproducibilidad`.
Salidas en [`resultados/`](resultados/): CSV crudos y resumidos, `entorno.txt` (versiones), `reproducibilidad.txt`
(hash SHA-256 idéntico con la misma semilla) y figuras PNG a 300 dpi.

## Reproducibilidad

Semilla base 42. Cada réplica usa `np.random.default_rng(SeedSequence([semilla, experimento, escenario, réplica]))`, por lo que
el resultado no depende del número de procesos. Entorno usado: ver [`resultados/entorno.txt`](resultados/entorno.txt).

## Resultados principales (media de 30 réplicas de 2 000 tiradas)

| Escenario | Exactitud Viterbi | Ingenuo ("siempre justo") | F1 HMM | F1 ventana χ² | Baum-Welch (aleatoria) | Baum-Welch (pegajosa) |
|---|---|---|---|---|---|---|
| E1 Base | 0.782 | 0.586 | 0.679 | 0.634 | 0.498 | 0.774 |
| E2 Sesgo debil | 0.626 | 0.617 | 0.043 | 0.202 | 0.353 | 0.459 |
| E3 Cambios frecuentes | 0.674 | 0.517 | 0.578 | 0.528 | 0.494 | 0.650 |
| E4 Sesgo fuerte | 0.949 | 0.602 | 0.937 | 0.849 | 0.941 | 0.947 |

Hallazgos: (1) con sesgo suficiente el HMM supera al ingenuo y a la ventana χ²; (2) con sesgo débil (E2) **no** supera al
ingenuo y la ventana detecta más; (3) Baum-Welch con inicialización aleatoria cae en óptimos locales, y una inicialización con
diagonal dominante lo iguala al modelo verdadero y converge ~4 veces más rápido; (4) perturbar A y B con ε ≤ 0,1 casi no afecta,
con ε = 0,8 la exactitud cae a ~0,56. Detalle, IC 95 % y pruebas estadísticas en el informe y en `resultados/*.csv`.

## Estructura

```
hmm.py            clase HMM
experimentos.py   escenarios, métricas, estadística, CLI
figuras.py        figuras (300 dpi)
tests/            pruebas (enumeración, fuerza bruta, hmmlearn, monotonía EM, reproducibilidad)
resultados/       CSV, figuras, entorno, hashes
```

## Limitaciones

Datos simulados con el mismo modelo que decodifica (techo optimista); parámetros de escenario supuestos; número de estados
conocido; la inicialización "pegajosa" incorpora conocimiento previo.
