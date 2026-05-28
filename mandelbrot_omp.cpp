/*
 * mandelbrot_omp.cpp  –  "Línea Base Paralela de la IA"  (OpenMP)
 *
 * Tarea A: Genera imagen 8K del conjunto de Mandelbrot con coloreo suave.
 * Tarea B: Aplica un filtro de desenfoque Gaussiano 2D (radio = 20, núcleo 41×41).
 *
 * Estrategia de paralelización:
 *  - Tarea A: #pragma omp parallel for schedule(static)
 *             Paralela sobre filas del fractal. Cada hilo calcula un bloque
 *             contiguo de filas asignado estáticamente.
 *  - Tarea B: #pragma omp parallel for schedule(static)
 *             Carga uniforme — todos los píxeles hacen (2r+1)² operaciones,
 *             por lo que la distribución estática es óptima.
 *
 *  Nota: la comparación empírica de planificadores (static / dynamic / guided)
 *        se encuentra en mandelbrot_sched_bench.cpp (Instrucción 3).
 *
 * Compilar:
 *   g++ -O2 -fopenmp -std=c++17 -o bin/mandelbrot_omp mandelbrot_omp.cpp
 *
 * Ejecutar con N hilos:
 *   OMP_NUM_THREADS=N ./bin/mandelbrot_omp
 *
 * Salida (carpeta output/):
 *   output/mandelbrot_8k.ppm          (fractal sin filtro)
 *   output/mandelbrot_8k_blurred.ppm  (fractal con Gaussiano)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <omp.h>
#include "Mandelbrot.hpp"
#include "ImageIO.hpp"

using Clock = std::chrono::high_resolution_clock;
using Sec   = std::chrono::duration<double>;

static constexpr int WIDTH        = 7680;
static constexpr int HEIGHT       = 4320;
static constexpr int MAX_ITER     = 1024;
static constexpr int GAUSS_RADIUS = 20;

static const Mandelbrot mb(WIDTH, HEIGHT, MAX_ITER);

// ---------------------------------------------------------------------------
// Tarea B – Convolución Gaussiana 2D paralela con OpenMP
// ---------------------------------------------------------------------------
static void gaussian_blur_omp(const std::vector<Pixel>& src,
                               std::vector<Pixel>&       dst,
                               const std::vector<double>& kernel,
                               int radius)
{
    const int size = 2 * radius + 1;

    #pragma omp parallel for schedule(static) default(none) \
            shared(src, dst, kernel, radius, size)
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double r = 0.0, g = 0.0, b = 0.0;
            for (int ky = -radius; ky <= radius; ++ky) {
                const int          ny   = std::clamp(y + ky, 0, HEIGHT - 1);
                const double*      krow = kernel.data() + (ky + radius) * size;
                const Pixel*       srow = src.data() + ny * WIDTH;
                for (int kx = -radius; kx <= radius; ++kx) {
                    const int    nx = std::clamp(x + kx, 0, WIDTH - 1);
                    const double w  = krow[kx + radius];
                    r += w * srow[nx].r;
                    g += w * srow[nx].g;
                    b += w * srow[nx].b;
                }
            }
            dst[y * WIDTH + x] = {
                static_cast<uint8_t>(std::clamp(r, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(g, 0.0, 255.0)),
                static_cast<uint8_t>(std::clamp(b, 0.0, 255.0))
            };
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::filesystem::create_directories("output");
    const int nthreads = omp_get_max_threads();

    std::cout << "============================================\n"
              << "  Mandelbrot + Gaussiano  [OpenMP — AI Baseline]\n"
              << "============================================\n"
              << "  Resolución  : " << WIDTH << " x " << HEIGHT << "\n"
              << "  Max iter    : " << MAX_ITER << "\n"
              << "  Radio Gauss : " << GAUSS_RADIUS
              << "  (núcleo " << 2*GAUSS_RADIUS+1 << "x" << 2*GAUSS_RADIUS+1 << ")\n"
              << "  Hilos OpenMP: " << nthreads << "\n\n";

    std::vector<Pixel> image(WIDTH * HEIGHT);
    std::vector<Pixel> blurred(WIDTH * HEIGHT);

    // ------------------------------------------------------------------
    // TAREA A  –  Generación paralela del fractal
    // ------------------------------------------------------------------
    std::cout << "[Tarea A] Generando fractal de Mandelbrot (OpenMP)...\n";
    auto ta0 = Clock::now();

    #pragma omp parallel for schedule(static) default(none) shared(image, mb)
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
            image[y * WIDTH + x] = mb.compute(x, y);

    double ta = Sec(Clock::now() - ta0).count();
    std::cout << "  Tiempo: " << std::fixed << std::setprecision(4) << ta << " s\n";
    ImageIO::save_ppm("output/mandelbrot_8k.ppm", image, WIDTH, HEIGHT);

    // ------------------------------------------------------------------
    // TAREA B  –  Desenfoque Gaussiano paralelo
    // ------------------------------------------------------------------
    std::cout << "\n[Tarea B] Aplicando desenfoque Gaussiano 2D (OpenMP)...\n";
    auto kernel = Mandelbrot::make_gaussian_kernel(GAUSS_RADIUS);

    auto tb0 = Clock::now();
    gaussian_blur_omp(image, blurred, kernel, GAUSS_RADIUS);
    double tb = Sec(Clock::now() - tb0).count();

    std::cout << "  Tiempo: " << std::fixed << std::setprecision(4) << tb << " s\n";
    ImageIO::save_ppm("output/mandelbrot_8k_blurred.ppm", blurred, WIDTH, HEIGHT);

    // ------------------------------------------------------------------
    // Resumen
    // ------------------------------------------------------------------
    std::cout << "\n--------------------------------------------\n"
              << "  Tarea A : " << std::fixed << std::setprecision(4) << ta << " s\n"
              << "  Tarea B : " << std::fixed << std::setprecision(4) << tb << " s\n"
              << "  TOTAL   : " << std::fixed << std::setprecision(4) << (ta + tb) << " s\n"
              << "  Hilos   : " << nthreads << "\n"
              << "--------------------------------------------\n";

    return 0;
}

