#!/usr/bin/env python3
"""
plot_scaling.py  –  Genera gráficas de escalabilidad OpenMP

Lee output/scaling_results.csv (producido por mandelbrot_scaling_bench)
y genera dos figuras en output/:

  output/plot_time_vs_threads.png   – Tiempo de ejecución vs. Núm. de hilos
  output/plot_speedup_vs_threads.png – Speedup vs. Núm. de hilos (+ speedup ideal)

Uso:
  python3 plot_scaling.py
"""

import csv
import os
import sys
import math
import matplotlib
matplotlib.use("Agg")          # renderizar sin pantalla
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Configuración visual
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "legend.fontsize":  11,
    "lines.linewidth":  2,
    "lines.markersize": 7,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

CSV_PATH = os.path.join("output", "scaling_results.csv")
OUT_TIME    = os.path.join("output", "plot_time_vs_threads.png")
OUT_SPEEDUP = os.path.join("output", "plot_speedup_vs_threads.png")

# ---------------------------------------------------------------------------
# Leer CSV
# ---------------------------------------------------------------------------
def load_csv(path: str):
    if not os.path.exists(path):
        sys.exit(f"ERROR: No se encontró '{path}'.\n"
                 "Ejecuta primero: ./bin/mandelbrot_scaling_bench")
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows

# ---------------------------------------------------------------------------
# Gráfica 1 – Tiempo de ejecución vs. Núm. de hilos
# ---------------------------------------------------------------------------
def plot_time(rows, n_logical: int):
    threads = [int(r["threads"])    for r in rows]
    t_a     = [r["time_a"]         for r in rows]
    t_b     = [r["time_b"]         for r in rows]
    t_tot   = [r["time_total"]     for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(threads, t_a,   "o-", color="#2196F3", label="Tarea A – Fractal")
    ax.plot(threads, t_b,   "s-", color="#FF9800", label="Tarea B – Gaussiano")
    ax.plot(threads, t_tot, "D-", color="#4CAF50", label="Total (A + B)")

    # Línea vertical en el número de núcleos lógicos
    ax.axvline(n_logical, color="red", linestyle=":", linewidth=1.5,
               label=f"Núcleos lógicos ({n_logical})")

    ax.set_xlabel("Número de hilos")
    ax.set_ylabel("Tiempo de ejecución (s)")
    ax.set_title("Tiempo de Ejecución vs. Número de Hilos\n"
                 "OpenMP – Mandelbrot 4K + Filtro Gaussiano")
    ax.set_xticks(threads)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_TIME)
    plt.close(fig)
    print(f"  Guardado: {OUT_TIME}")

# ---------------------------------------------------------------------------
# Gráfica 2 – Speedup vs. Núm. de hilos
# ---------------------------------------------------------------------------
def plot_speedup(rows, n_logical: int):
    threads = [int(r["threads"])    for r in rows]
    s_a     = [r["speedup_a"]      for r in rows]
    s_b     = [r["speedup_b"]      for r in rows]
    s_tot   = [r["speedup_total"]  for r in rows]
    ideal   = [float(t)            for t in threads]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(threads, ideal, "--", color="gray",    linewidth=1.5, label="Speedup ideal (lineal)")
    ax.plot(threads, s_a,   "o-", color="#2196F3", label="Tarea A – Fractal")
    ax.plot(threads, s_b,   "s-", color="#FF9800", label="Tarea B – Gaussiano")
    ax.plot(threads, s_tot, "D-", color="#4CAF50", label="Total (A + B)")

    # Línea vertical en el número de núcleos lógicos
    ax.axvline(n_logical, color="red", linestyle=":", linewidth=1.5,
               label=f"Núcleos lógicos ({n_logical})")

    # Amdahl teórico aproximado si queremos mostrarlo (comentado por defecto)
    # f_par = 0.95  # fracción paralelizable estimada
    # amdahl = [1 / ((1 - f_par) + f_par / t) for t in threads]
    # ax.plot(threads, amdahl, "-.", color="purple", linewidth=1.2, label="Ley de Amdahl (f=0.95)")

    ax.set_xlabel("Número de hilos")
    ax.set_ylabel("Speedup  S(p) = T(1) / T(p)")
    ax.set_title("Aceleración (Speedup) vs. Número de Hilos\n"
                 "OpenMP – Mandelbrot 4K + Filtro Gaussiano")
    ax.set_xticks(threads)
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    # Límite Y empieza en 0 con un pequeño margen
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_SPEEDUP)
    plt.close(fig)
    print(f"  Guardado: {OUT_SPEEDUP}")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs("output", exist_ok=True)
    rows = load_csv(CSV_PATH)

    if not rows:
        sys.exit("ERROR: El CSV está vacío.")

    # Número de núcleos lógicos: la mitad del rango barrido
    max_threads = int(rows[-1]["threads"])
    n_logical   = max_threads // 2

    print(f"Datos cargados: {len(rows)} configuraciones "
          f"(1 .. {max_threads} hilos, {n_logical} núcleos lógicos)\n")

    plot_time(rows, n_logical)
    plot_speedup(rows, n_logical)

    print("\nListo.")

if __name__ == "__main__":
    main()
