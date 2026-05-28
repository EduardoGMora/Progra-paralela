/*
 * mandelbrot_omp.cpp  –  "Línea Base Paralela de la IA"  (OpenMP)
 *
 * Tarea A: Genera imagen 8K del conjunto de Mandelbrot con coloreo suave.
 * Tarea B: Aplica un filtro de desenfoque Gaussiano 2D (radio = 20, núcleo 41×41).
 *
 * Estrategia de paralelización:
 *  - Tarea A: Primero se mide con schedule(static) como referencia, luego con
 *             schedule(dynamic, 8) que resultó ser el planificador óptimo según
 *             el benchmark empírico (mandelbrot_sched_bench.cpp) en un Ryzen 5 3600:
 *
 *               Planificador  Chunk   Tiempo(s)   Speedup
 *               static        -       0.5415      1.000  ← referencia
 *               dynamic       8       0.3089      1.753x ← ÓPTIMO
 *               dynamic       4       0.3102      1.745x
 *               guided        -       0.3920      1.381x
 *
 *             El planificador dinámico compensa el desbalance de carga inherente
 *             al fractal (píxeles dentro del conjunto requieren MAX_ITER iteraciones
 *             mientras que los exteriores escapan rápidamente).
 *  - Tarea B: `#pragma omp parallel for schedule(static)`
 *             La carga es uniforme (todos los píxeles hacen exactamente
 *             (2r+1)² multiplicaciones), por lo que static es óptimo.
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
#include <omp.h>
#include "Mandelbrot.hpp"
#include "ImageIO.hpp"

// ---------------------------------------------------------------------------
// Parámetros globales  (idénticos a la versión secuencial)
// ---------------------------------------------------------------------------
static constexpr int    WIDTH        = 7680;
static constexpr int    HEIGHT       = 4320;
static constexpr int    MAX_ITER     = 1024;
static constexpr int    GAUSS_RADIUS = 20;   // núcleo 41×41

static const Mandelbrot mb(WIDTH, HEIGHT, MAX_ITER);

// ---------------------------------------------------------------------------
// TAREA A – Auxiliar: genera el fractal con schedule(static) para referencia
// ---------------------------------------------------------------------------
static void generate_mandelbrot_static(std::vector<Pixel>& img) {
    #pragma omp parallel for schedule(static) default(none) shared(img, mb)
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
            img[y * WIDTH + x] = mb.compute(x, y);
}

// ---------------------------------------------------------------------------
// TAREA B – Convolución Gaussiana 2D  *** PARALELIZADA ***
//
//  Cada hilo trabaja sobre filas independientes → no hay condiciones de carrera.
//  El kernel y src son de solo lectura → no requieren sincronización.
//  schedule(static) es óptimo porque la carga por fila es uniforme.
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
                int ny = std::clamp(y + ky, 0, HEIGHT - 1);
                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = std::clamp(x + kx, 0, WIDTH - 1);
                    double w = kernel[(ky + radius) * size + (kx + radius)];
                    const Pixel& p = src[ny * WIDTH + nx];
                    r += w * p.r;
                    g += w * p.g;
                    b += w * p.b;
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
    using Clock = std::chrono::high_resolution_clock;
    using Sec   = std::chrono::duration<double>;

    const int nthreads = omp_get_max_threads();

    std::cout << "============================================\n"
              << "  Mandelbrot + Gaussiano  [OpenMP]\n"
              << "============================================\n"
              << "  Resolución  : " << WIDTH << " x " << HEIGHT << "\n"
              << "  Max iter    : " << MAX_ITER << "\n"
              << "  Radio Gauss : " << GAUSS_RADIUS
              << "  (núcleo " << 2*GAUSS_RADIUS+1 << "x" << 2*GAUSS_RADIUS+1 << ")\n"
              << "  Hilos OpenMP: " << nthreads << "\n\n";

    std::vector<Pixel> image(WIDTH * HEIGHT);
    std::vector<Pixel> blurred(WIDTH * HEIGHT);

    // ------------------------------------------------------------------
    // TAREA A  –  Comparación de planificadores
    // ------------------------------------------------------------------
    std::cout << "[Tarea A] Paso 1: schedule(static) — referencia...\n";
    auto ta_s0 = Clock::now();
    generate_mandelbrot_static(image);
    double ta_static = Sec(Clock::now() - ta_s0).count();
    std::cout << "  Tiempo static    : " << ta_static << " s\n";

    std::cout << "[Tarea A] Paso 2: schedule(dynamic, 8) — óptimo...\n";
    auto ta0 = Clock::now();

    #pragma omp parallel for schedule(dynamic, 8) default(none) shared(image, mb)
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
            image[y * WIDTH + x] = mb.compute(x, y);

    double ta = Sec(Clock::now() - ta0).count();
    std::cout << "  Tiempo dynamic,8 : " << ta << " s"
              << "  (speedup " << std::fixed << std::setprecision(3)
              << (ta_static / ta) << "x vs static)\n";
    ImageIO::save_ppm("output/mandelbrot_8k.ppm", image, WIDTH, HEIGHT);

    // ------------------------------------------------------------------
    // TAREA B  –  schedule(static)
    // ------------------------------------------------------------------
    std::cout << "\n[Tarea B] Aplicando Gaussiano  (schedule=static)...\n";
    auto kernel = Mandelbrot::make_gaussian_kernel(GAUSS_RADIUS);

    auto tb0 = Clock::now();
    gaussian_blur_omp(image, blurred, kernel, GAUSS_RADIUS);
    double tb = Sec(Clock::now() - tb0).count();

    std::cout << "  Tiempo: " << tb << " s\n";
    ImageIO::save_ppm("output/mandelbrot_8k_blurred.ppm", blurred, WIDTH, HEIGHT);

    // ------------------------------------------------------------------
    // Resumen
    // ------------------------------------------------------------------
    std::cout << "\n--------------------------------------------\n"
              << "  Tarea A static   : " << ta_static << " s\n"
              << "  Tarea A dynamic,8: " << ta << " s"
              << "  (speedup " << std::fixed << std::setprecision(3)
              << (ta_static / ta) << "x)\n"
              << "  Tarea B : " << tb << " s\n"
              << "  TOTAL   : " << (ta + tb) << " s\n"
              << "  Hilos   : " << nthreads << "\n"
              << "--------------------------------------------\n";

    return 0;
}

