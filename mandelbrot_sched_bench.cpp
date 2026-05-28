/*
 * mandelbrot_sched_bench.cpp  –  Benchmark de planificadores OpenMP (Tarea A)
 *
 * Evalúa schedule(static), schedule(dynamic,C) y schedule(guided,C)
 * con distintos chunk sizes sobre la generación del fractal Mandelbrot.
 *
 * Resolución 3840×2160 (4K) y MAX_ITER=512 para mantener el desbalance
 * de carga real del fractal con tiempo de benchmark razonable.
 *
 * Compilar:
 *   g++ -O2 -fopenmp -std=c++17 -o bin/mandelbrot_sched_bench mandelbrot_sched_bench.cpp
 *
 * Ejecutar:
 *   OMP_NUM_THREADS=12 ./bin/mandelbrot_sched_bench
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>
#include <limits>
#include <omp.h>
#include "Mandelbrot.hpp"

static constexpr int W        = 3840;
static constexpr int H        = 2160;
static constexpr int MAX_ITER = 512;
static constexpr int RUNS     = 3;

static const Mandelbrot mb(W, H, MAX_ITER);

// ---------------------------------------------------------------------------
// Kernels con planificador fijo en tiempo de compilación
// ---------------------------------------------------------------------------

static void kern_static(std::vector<Pixel>& img) {
    #pragma omp parallel for schedule(static) default(none) shared(img, mb)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            img[y*W+x] = mb.compute(x, y);
}

// --- dynamic ---
static void kern_dyn(std::vector<Pixel>& img, int chunk=1) {
    #pragma omp parallel for schedule(dynamic,chunk) default(none) shared(img, mb) firstprivate(chunk)
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) img[y*W+x] = mb.compute(x,y);
}

// --- guided ---
static void kern_guid_def(std::vector<Pixel>& img, int /*chunk*/) {
    #pragma omp parallel for schedule(guided) default(none) shared(img, mb)
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) img[y*W+x] = mb.compute(x,y);
}
static void kern_guid(std::vector<Pixel>& img, int chunk=1) {
    #pragma omp parallel for schedule(guided,chunk) default(none) shared(img, mb) firstprivate(chunk)
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) img[y*W+x] = mb.compute(x,y);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;
using Sec   = std::chrono::duration<double>;

template<typename Fn>
static double bench(Fn&& fn) {
    double best = std::numeric_limits<double>::max();
    for (int r = 0; r < RUNS; ++r) {
        auto t0 = Clock::now();
        fn();
        double t = Sec(Clock::now() - t0).count();
        if (t < best) best = t;
    }
    return best;
}

int main() {
    const int nthreads = omp_get_max_threads();
    std::vector<Pixel> img(W * H);

    std::cout << "=============================================================\n"
              << "  Benchmark Planificadores OpenMP  –  Tarea A  Mandelbrot\n"
              << "=============================================================\n"
              << "  Resolución : " << W << " x " << H
              << "  |  Max iter: " << MAX_ITER
              << "  |  Hilos: " << nthreads
              << "  |  Runs: " << RUNS << "\n"
              << "-------------------------------------------------------------\n"
              << std::left
              << std::setw(14) << "Planificador"
              << std::setw(10) << "Chunk"
              << std::setw(14) << "Tiempo (s)"
              << "Speedup vs static\n"
              << "-------------------------------------------------------------\n";

    // ---- Referencia: static (por defecto) ----
    double t_ref = bench([&]{ kern_static(img); });
    std::cout << std::setw(14) << "static"
              << std::setw(10) << "-"
              << std::fixed << std::setprecision(4) << std::setw(14) << t_ref
              << "1.000  ← referencia\n";

    struct Config {
        const char* name;
        int         chunk;   // 0 = default
        void (*fn)(std::vector<Pixel>&, int);
    };

    Config configs[] = {
        {"dynamic",   1,   kern_dyn },
        {"dynamic",   2,   kern_dyn },
        {"dynamic",   4,   kern_dyn },
        {"dynamic",   8,   kern_dyn },
        {"dynamic",   16,  kern_dyn },
        {"dynamic",   32,  kern_dyn },
        {"dynamic",   64,  kern_dyn },
        {"dynamic",   128, kern_dyn },
        {"guided",    0,   kern_guid_def },
        {"guided",    1,   kern_guid },
        {"guided",    2,   kern_guid },
        {"guided",    4,   kern_guid },
        {"guided",    8,   kern_guid },
        {"guided",    16,  kern_guid },
        {"guided",    32,  kern_guid },
        {"guided",    64,  kern_guid },
        {"guided",    128, kern_guid },
    };

    double      best_t     = t_ref;
    const char* best_name  = "static";
    int         best_chunk = -1;

    for (auto& c : configs) {
        double t       = bench([&]{ c.fn(img, c.chunk); });
        double speedup = t_ref / t;

        std::string chunk_str = (c.chunk == 0) ? "-" : std::to_string(c.chunk);
        std::cout << std::setw(14) << c.name
                  << std::setw(10) << chunk_str
                  << std::fixed << std::setprecision(4) << std::setw(14) << t
                  << std::fixed << std::setprecision(3) << speedup << "x\n";

        if (t < best_t) {
            best_t     = t;
            best_name  = c.name;
            best_chunk = c.chunk;
        }
    }

    std::cout << "-------------------------------------------------------------\n"
              << "  ÓPTIMO: schedule(" << best_name;
    if (best_chunk > 0) std::cout << ", " << best_chunk;
    std::cout << ")"
              << "  →  " << std::fixed << std::setprecision(4) << best_t << " s"
              << "  (speedup " << std::fixed << std::setprecision(3)
              << (t_ref / best_t) << "x vs static)\n"
              << "=============================================================\n";

    return 0;
}
