/*
 * mandelbrot_scaling_bench.cpp  –  Benchmark de escalabilidad OpenMP
 *
 * Mide el tiempo de Tarea A (fractal) y Tarea B (Gaussiano) para cada
 * número de hilos desde 1 hasta 2 × núcleos lógicos.
 *
 * Salida:
 *   output/scaling_results.csv   (leído por plot_scaling.py)
 *
 * Compilar:
 *   g++ -O2 -fopenmp -std=c++17 -o bin/mandelbrot_scaling_bench mandelbrot_scaling_bench.cpp
 *
 * Ejecutar:
 *   ./bin/mandelbrot_scaling_bench
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <limits>
#include <filesystem>
#include <omp.h>
#include "Mandelbrot.hpp"

// ---------------------------------------------------------------------------
// Parámetros del benchmark (reducidos respecto a 8K para que sea rápido)
// ---------------------------------------------------------------------------
static constexpr int W           = 3840;
static constexpr int H           = 2160;
static constexpr int MAX_ITER    = 512;
static constexpr int GAUSS_RADIUS = 10;   // núcleo 21×21
static constexpr int RUNS        = 3;     // repeticiones; se guarda el mínimo

static const Mandelbrot mb(W, H, MAX_ITER);

using Clock = std::chrono::high_resolution_clock;
using Sec   = std::chrono::duration<double>;

// ---------------------------------------------------------------------------
// Kernels paralelizados — número de hilos fijado antes de llamarlos con
// omp_set_num_threads().
// ---------------------------------------------------------------------------

static double bench_task_a(std::vector<Pixel>& img) {
    double best = std::numeric_limits<double>::max();
    for (int r = 0; r < RUNS; ++r) {
        auto t0 = Clock::now();
        #pragma omp parallel for schedule(static) default(none) shared(img, mb)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                img[y * W + x] = mb.compute(x, y);
        double t = Sec(Clock::now() - t0).count();
        if (t < best) best = t;
    }
    return best;
}

static double bench_task_b(const std::vector<Pixel>& src,
                            std::vector<Pixel>&       dst,
                            const std::vector<double>& kernel)
{
    const int size = 2 * GAUSS_RADIUS + 1;
    double best = std::numeric_limits<double>::max();
    for (int r = 0; r < RUNS; ++r) {
        auto t0 = Clock::now();
        #pragma omp parallel for schedule(static) default(none) \
                shared(src, dst, kernel, size)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                double rv = 0.0, gv = 0.0, bv = 0.0;
                for (int ky = -GAUSS_RADIUS; ky <= GAUSS_RADIUS; ++ky) {
                    const int     ny   = std::clamp(y + ky, 0, H - 1);
                    const double* krow = kernel.data() + (ky + GAUSS_RADIUS) * size;
                    const Pixel*  srow = src.data() + ny * W;
                    for (int kx = -GAUSS_RADIUS; kx <= GAUSS_RADIUS; ++kx) {
                        const int    nx = std::clamp(x + kx, 0, W - 1);
                        const double w  = krow[kx + GAUSS_RADIUS];
                        rv += w * srow[nx].r;
                        gv += w * srow[nx].g;
                        bv += w * srow[nx].b;
                    }
                }
                dst[y * W + x] = {
                    static_cast<uint8_t>(std::clamp(rv, 0.0, 255.0)),
                    static_cast<uint8_t>(std::clamp(gv, 0.0, 255.0)),
                    static_cast<uint8_t>(std::clamp(bv, 0.0, 255.0))
                };
            }
        }
        double t = Sec(Clock::now() - t0).count();
        if (t < best) best = t;
    }
    return best;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::filesystem::create_directories("output");

    const int max_logical = omp_get_max_threads();
    const int max_threads = 2 * max_logical;

    std::cout << "============================================================\n"
              << "  Benchmark de Escalabilidad OpenMP  –  Mandelbrot\n"
              << "============================================================\n"
              << "  Resolución   : " << W << " x " << H << "\n"
              << "  Max iter     : " << MAX_ITER << "\n"
              << "  Radio Gauss  : " << GAUSS_RADIUS
              << "  (núcleo " << 2*GAUSS_RADIUS+1 << "x" << 2*GAUSS_RADIUS+1 << ")\n"
              << "  Núcleos log. : " << max_logical << "\n"
              << "  Rango hilos  : 1 .. " << max_threads << "\n"
              << "  Runs/config  : " << RUNS << "  (se reporta el mínimo)\n"
              << "------------------------------------------------------------\n"
              << std::left
              << std::setw(8)  << "Hilos"
              << std::setw(14) << "T_A (s)"
              << std::setw(14) << "T_B (s)"
              << std::setw(14) << "T_total (s)"
              << std::setw(10) << "S_A"
              << std::setw(10) << "S_B"
              << "S_total\n"
              << "------------------------------------------------------------\n";

    std::vector<Pixel> img(W * H), blurred(W * H);
    auto kernel = Mandelbrot::make_gaussian_kernel(GAUSS_RADIUS);

    // Medición secuencial (1 hilo) como referencia
    omp_set_num_threads(1);
    double t1_a = bench_task_a(img);
    double t1_b = bench_task_b(img, blurred, kernel);
    double t1   = t1_a + t1_b;

    // CSV
    std::ofstream csv("output/scaling_results.csv");
    csv << "threads,time_a,time_b,time_total,speedup_a,speedup_b,speedup_total\n";

    for (int t = 1; t <= max_threads; ++t) {
        omp_set_num_threads(t);

        double ta    = bench_task_a(img);
        double tb    = bench_task_b(img, blurred, kernel);
        double total = ta + tb;

        double sa = t1_a / ta;
        double sb = t1_b / tb;
        double st = t1   / total;

        std::cout << std::setw(8)  << t
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << ta
                  << std::setw(14) << tb
                  << std::setw(14) << total
                  << std::setw(10) << sa
                  << std::setw(10) << sb
                  << st << "\n";

        csv << t << ","
            << ta << "," << tb << "," << total << ","
            << sa << "," << sb << "," << st << "\n";
    }

    csv.close();

    std::cout << "------------------------------------------------------------\n"
              << "  Referencia (1 hilo): T_A=" << t1_a << " s  T_B=" << t1_b
              << " s  TOTAL=" << t1 << " s\n"
              << "  CSV guardado: output/scaling_results.csv\n"
              << "============================================================\n";
    return 0;
}
