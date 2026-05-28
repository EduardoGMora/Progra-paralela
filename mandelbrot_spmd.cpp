/*
 * mandelbrot_spmd.cpp  –  Tarea 5: SPMD y Afinidad
 *
 * ─── Vectorización SPMD ────────────────────────────────────────────────────
 *
 * Restructura el filtro Gaussiano de Tarea B en estructura SPMD:
 *   · Nivel hilo  : #pragma omp parallel for  → hilos dividen filas Y
 *   · Nivel SIMD  : #pragma omp simd reduction(+:r,g,b) → SIMD en bucle kx
 *                   Genera instrucciones AVX2/SSE en el bucle de kernel.
 *
 * El bucle kx se divide en dos ramas:
 *   · Interior (x ∈ [radius, W-radius)): kx+x siempre válido → sin clamp
 *     → completamente vectorizable
 *   · Borde: fallback escalar con std::clamp (solo 2·radius píxeles/fila)
 *
 * Verificar vectorización:
 *   Compilar con -fopt-info-vec (ver Makefile target 'spmd').
 *   El compilador imprime cada bucle que vectorizó con AVX2.
 *
 * ─── Afinidad de Hilos (10 pts extra) ──────────────────────────────────────
 *
 * El programa reporta el modo OMP_PROC_BIND activo y el lugar de cada hilo.
 * Ejecutar con distintos ajustes para comparar impacto en caché L1/L2:
 *
 *   # Sin afinidad (por defecto):
 *   OMP_NUM_THREADS=6 ./bin/mandelbrot_spmd
 *
 *   # Hilos agrupados en núcleos físicos contiguos (mejor localidad L2):
 *   OMP_PROC_BIND=close  OMP_PLACES=cores   OMP_NUM_THREADS=6 ./bin/mandelbrot_spmd
 *
 *   # Hilos distribuidos en todos los núcleos físicos (mejor ancho de banda):
 *   OMP_PROC_BIND=spread OMP_PLACES=cores   OMP_NUM_THREADS=6 ./bin/mandelbrot_spmd
 *
 *   # Usar hilos lógicos (SMT), uno por socket lógico:
 *   OMP_PROC_BIND=close  OMP_PLACES=threads OMP_NUM_THREADS=12 ./bin/mandelbrot_spmd
 *
 * ─── Escalado de Hilos (datos para gráficas del reporte) ───────────────────
 *
 * Genera tabla Tiempo-vs-Hilos y Speedup-vs-Hilos de 1 hasta 2×nproc.
 *
 * Compilar:
 *   make spmd
 *   (equivale a: g++ -O3 -march=native -fopenmp -std=c++17 -fopt-info-vec
 *                -o bin/mandelbrot_spmd mandelbrot_spmd.cpp)
 *
 * Ejecutar:
 *   OMP_NUM_THREADS=12 ./bin/mandelbrot_spmd
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include "Mandelbrot.hpp"

using Clock = std::chrono::high_resolution_clock;
using Sec   = std::chrono::duration<double>;

// Resolución 4K para el benchmark de escalado (razonable en tiempo)
static constexpr int W        = 3840;
static constexpr int H        = 2160;
static constexpr int MAX_ITER = 1024;
static constexpr int RADIUS   = 20;

static const Mandelbrot mb(W, H, MAX_ITER);

// ---------------------------------------------------------------------------
// Blur de referencia: OpenMP paralelo SIN pragma simd
// (misma estructura que mandelbrot_omp.cpp, sirve de baseline)
// ---------------------------------------------------------------------------
static void blur_omp_baseline(const std::vector<Pixel>& src,
                               std::vector<Pixel>&       dst,
                               const std::vector<double>& kernel,
                               int radius)
{
    const int size = 2 * radius + 1;
    #pragma omp parallel for schedule(static) default(none) \
            shared(src, dst, kernel, radius, size)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double r = 0.0, g = 0.0, b = 0.0;
            for (int ky = -radius; ky <= radius; ++ky) {
                int ny = std::clamp(y + ky, 0, H - 1);
                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = std::clamp(x + kx, 0, W - 1);
                    const double w = kernel[(ky + radius) * size + (kx + radius)];
                    r += w * src[ny * W + nx].r;
                    g += w * src[ny * W + nx].g;
                    b += w * src[ny * W + nx].b;
                }
            }
            dst[y * W + x] = {
                static_cast<uint8_t>(std::clamp(r, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(g, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(b, 0.0, 255.0))
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Blur SPMD: OpenMP paralelo + SIMD explícito en el bucle de kernel
//
// Estructura SPMD:
//   Nivel 1 (MIMD): #pragma omp parallel for → cada hilo procesa un rango de filas
//   Nivel 2 (SIMD): #pragma omp simd reduction(+:r,g,b) → vectoriza el bucle kx
//
// División interior/borde:
//   Para x ∈ [radius, W-radius): x+kx ∈ [0, W-1] siempre → sin std::clamp
//   → el compilador puede emitir instrucciones AVX2 para el bucle kx.
//   Los ~2·radius píxeles en cada borde lateral usan el camino escalar.
// ---------------------------------------------------------------------------
static void blur_spmd(const std::vector<Pixel>& src,
                      std::vector<Pixel>&        dst,
                      const std::vector<double>& kernel,
                      int radius)
{
    const int size = 2 * radius + 1;

    #pragma omp parallel for schedule(static) default(none) \
            shared(src, dst, kernel, radius, size)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double r = 0.0, g = 0.0, b = 0.0;

            for (int ky = -radius; ky <= radius; ++ky) {
                const int    ny   = std::clamp(y + ky, 0, H - 1);
                const double* krow = kernel.data() + (ky + radius) * size;
                const Pixel*  srow = src.data() + ny * W;

                if (x >= radius && x < W - radius) {
                    // Camino interior: sin clamp → permite vectorización SIMD
                    #pragma omp simd reduction(+:r, g, b)
                    for (int kx = -radius; kx <= radius; ++kx) {
                        const double w = krow[kx + radius];
                        r += w * srow[x + kx].r;
                        g += w * srow[x + kx].g;
                        b += w * srow[x + kx].b;
                    }
                } else {
                    // Camino borde: fallback escalar con clamp
                    for (int kx = -radius; kx <= radius; ++kx) {
                        const int    nx = std::clamp(x + kx, 0, W - 1);
                        const double w  = krow[kx + radius];
                        r += w * srow[nx].r;
                        g += w * srow[nx].g;
                        b += w * srow[nx].b;
                    }
                }
            }
            dst[y * W + x] = {
                static_cast<uint8_t>(std::clamp(r, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(g, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(b, 0.0, 255.0))
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: mide el mínimo de RUNS ejecuciones de fn()
// ---------------------------------------------------------------------------
static constexpr int BENCH_RUNS = 3;

template<typename Fn>
static double bench_min(Fn&& fn) {
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < BENCH_RUNS; ++i) {
        auto t0 = Clock::now();
        fn();
        double t = Sec(Clock::now() - t0).count();
        if (t < best) best = t;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Reporte de afinidad de hilos
//
// Muestra: OMP_PROC_BIND activo, número de lugares (places), procesadores
// por lugar y qué lugar ocupa cada hilo en la región paralela actual.
// ---------------------------------------------------------------------------
static void report_affinity(int nthreads) {
    // omp_get_proc_bind() devuelve el valor de OMP_PROC_BIND
    omp_proc_bind_t bind = omp_get_proc_bind();
    const char* bind_str = "desconocido";
    switch (bind) {
        case omp_proc_bind_false:  bind_str = "false  (sin afinidad)";  break;
        case omp_proc_bind_true:   bind_str = "true";                   break;
        case omp_proc_bind_master: bind_str = "master";                 break;
        case omp_proc_bind_close:  bind_str = "close  ← hilos agrupados"; break;
        case omp_proc_bind_spread: bind_str = "spread ← hilos distribuidos"; break;
    }

    const int nplaces = omp_get_num_places();
    std::cout << "  OMP_PROC_BIND : " << bind_str << "\n"
              << "  OMP_PLACES    : " << nplaces << " lugar(es) disponibles\n";

    // Mostrar procesadores lógicos por lugar (hasta 8 para no saturar la salida)
    for (int p = 0; p < std::min(nplaces, 8); ++p)
        std::cout << "    lugar " << p << " → "
                  << omp_get_place_num_procs(p) << " proc(s)\n";

    // Mostrar en qué lugar queda cada hilo
    std::cout << "  Asignación de hilos:\n";
    omp_set_num_threads(nthreads);
    #pragma omp parallel default(none) shared(std::cout)
    {
        #pragma omp critical
        std::cout << "    hilo " << std::setw(2) << omp_get_thread_num()
                  << " → lugar " << omp_get_place_num() << "\n";
    }
}

// ---------------------------------------------------------------------------
// Escalado de hilos
//
// Evalúa blur_spmd con 1..2×max_threads hilos.
// Imprime tabla lista para copiar en una hoja de cálculo / gnuplot.
// Provee los datos para las Gráficas 3 y 4 del reporte técnico:
//   · Gráfica 3: Tiempo de Ejecución vs. Número de Hilos
//   · Gráfica 4: Speedup vs. Número de Hilos
// ---------------------------------------------------------------------------
static void thread_scaling(const std::vector<Pixel>& image,
                            const std::vector<Pixel>& /*dst_placeholder*/,
                            const std::vector<double>& kernel)
{
    const int max_t  = omp_get_max_threads();
    const int test_t = max_t * 2;  // hasta el doble de núcleos lógicos

    std::vector<Pixel> dst(image.size());

    std::cout << "\n[Escalado de Hilos] Blur SPMD | "
              << W << "x" << H << " | radio=" << RADIUS
              << " | runs=" << BENCH_RUNS << "\n"
              << std::left
              << std::setw(8)  << "Hilos"
              << std::setw(14) << "Tiempo (s)"
              << std::setw(12) << "Speedup"
              << "Eficiencia\n"
              << std::string(46, '-') << "\n";

    double t1 = 0.0;
    for (int nt = 1; nt <= test_t; ++nt) {
        omp_set_num_threads(nt);
        // Una sola medición por conteo de hilos para mantener el benchmark ágil
        auto t0 = Clock::now();
        blur_spmd(image, dst, kernel, RADIUS);
        double t = Sec(Clock::now() - t0).count();
        if (nt == 1) t1 = t;

        double speedup    = t1 / t;
        double efficiency = speedup / nt * 100.0;

        std::cout << std::left
                  << std::setw(8)  << nt
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << t
                  << std::setprecision(3)
                  << std::setw(12) << speedup
                  << std::setprecision(1) << efficiency << " %\n";
    }
    std::cout << std::string(46, '-') << "\n"
              << "  Núcleos lógicos del sistema: " << max_t << "\n"
              << "  Límite teórico (Amdahl): ver punto donde la eficiencia cae\n"
              << "  bruscamente (típicamente > " << max_t << " hilos).\n";

    // Restaurar el número de hilos a lo que indica OMP_NUM_THREADS
    omp_set_num_threads(max_t);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const int nthreads = omp_get_max_threads();

    std::cout << "============================================\n"
              << "  Tarea 5: SPMD y Afinidad  [OpenMP]\n"
              << "============================================\n"
              << "  Resolución  : " << W << " x " << H << " (4K)\n"
              << "  Max iter    : " << MAX_ITER << "\n"
              << "  Radio Gauss : " << RADIUS
              << "  (núcleo " << 2*RADIUS+1 << "x" << 2*RADIUS+1 << ")\n"
              << "  Hilos OpenMP: " << nthreads << "\n\n";

    // ---- 1. Afinidad de hilos ----
    std::cout << "[Afinidad] Configuración actual:\n";
    report_affinity(nthreads);
    std::cout << "\n  Para cambiar la afinidad, ejecutar como:\n"
              << "    OMP_PROC_BIND=close  OMP_PLACES=cores   OMP_NUM_THREADS="
              << nthreads << " ./bin/mandelbrot_spmd\n"
              << "    OMP_PROC_BIND=spread OMP_PLACES=cores   OMP_NUM_THREADS="
              << nthreads << " ./bin/mandelbrot_spmd\n"
              << "    OMP_PROC_BIND=close  OMP_PLACES=threads OMP_NUM_THREADS="
              << nthreads << " ./bin/mandelbrot_spmd\n\n";

    // ---- 2. Generar fractal ----
    std::cout << "[Generación] Fractal Mandelbrot 4K...\n";
    std::vector<Pixel> image(W * H);
    {
        auto t0 = Clock::now();
        #pragma omp parallel for schedule(dynamic, 8) default(none) shared(image, mb)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                image[y * W + x] = mb.compute(x, y);
        std::cout << "  Tiempo: " << std::fixed << std::setprecision(4)
                  << Sec(Clock::now() - t0).count() << " s\n\n";
    }

    // ---- 3. Comparar baseline OMP vs SPMD/SIMD ----
    auto kernel = Mandelbrot::make_gaussian_kernel(RADIUS);
    std::vector<Pixel> dst(W * H);

    std::cout << "[Comparación] Blur OMP baseline vs SPMD/SIMD (" << nthreads << " hilos)\n";

    double t_base = bench_min([&]{ blur_omp_baseline(image, dst, kernel, RADIUS); });
    std::cout << "  OMP sin simd : " << std::fixed << std::setprecision(4)
              << t_base << " s\n";

    double t_spmd = bench_min([&]{ blur_spmd(image, dst, kernel, RADIUS); });
    std::cout << "  SPMD + simd  : " << std::fixed << std::setprecision(4)
              << t_spmd << " s"
              << "  (speedup " << std::fixed << std::setprecision(3)
              << (t_base / t_spmd) << "x)\n\n";

    std::cout << "  Nota: si el speedup SPMD es ~1x, el compilador ya optimizaba\n"
              << "  el bucle internamente con -O3. Usar -fopt-info-vec para\n"
              << "  confirmar qué bucles se vectorizaron en ambas funciones.\n\n";

    // ---- 4. Escalado de hilos (datos para gráficas del reporte) ----
    thread_scaling(image, dst, kernel);

    return 0;
}
