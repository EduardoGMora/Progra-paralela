/*
 * mandelbrot_seq.cpp  –  Versión SECUENCIAL
 *
 * Tarea A: Genera imagen 8K del conjunto de Mandelbrot con coloreo suave.
 * Tarea B: Aplica un filtro de desenfoque Gaussiano 2D (radio = 20, núcleo 41×41).
 *
 * Compilar:
 *   g++ -O2 -std=c++17 -o bin/mandelbrot_seq mandelbrot_seq.cpp
 *
 * Salida (carpeta output/):
 *   output/mandelbrot_8k.ppm          (fractal sin filtro)
 *   output/mandelbrot_8k_blurred.ppm  (fractal con Gaussiano)
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include "Mandelbrot.hpp"
#include "ImageIO.hpp"

// ---------------------------------------------------------------------------
// Parámetros globales
// ---------------------------------------------------------------------------
static constexpr int WIDTH        = 7680;
static constexpr int HEIGHT       = 4320;
static constexpr int MAX_ITER     = 1024;
static constexpr int GAUSS_RADIUS = 20;   // núcleo 41×41

static const Mandelbrot mb(WIDTH, HEIGHT, MAX_ITER);

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    using Clock = std::chrono::high_resolution_clock;
    using Sec   = std::chrono::duration<double>;

    std::cout << "============================================\n"
              << "  Mandelbrot + Gaussiano  [SECUENCIAL]\n"
              << "============================================\n"
              << "  Resolución  : " << WIDTH << " x " << HEIGHT << "\n"
              << "  Max iter    : " << MAX_ITER << "\n"
              << "  Radio Gauss : " << GAUSS_RADIUS
              << "  (núcleo " << 2*GAUSS_RADIUS+1 << "x" << 2*GAUSS_RADIUS+1 << ")\n\n";

    std::filesystem::create_directories("output");

    std::vector<Pixel> image(WIDTH * HEIGHT);
    std::vector<Pixel> blurred(WIDTH * HEIGHT);

    // ------------------------------------------------------------------
    // TAREA A
    // ------------------------------------------------------------------
    std::cout << "[Tarea A] Generando fractal de Mandelbrot...\n";
    auto ta0 = Clock::now();

    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
            image[y * WIDTH + x] = mb.compute(x, y);

    double ta = Sec(Clock::now() - ta0).count();
    std::cout << "  Tiempo: " << ta << " s\n";
    ImageIO::save_ppm("output/mandelbrot_8k.ppm", image, WIDTH, HEIGHT);

    // ------------------------------------------------------------------
    // TAREA B
    // ------------------------------------------------------------------
    std::cout << "\n[Tarea B] Aplicando desenfoque Gaussiano 2D...\n";
    auto kernel = Mandelbrot::make_gaussian_kernel(GAUSS_RADIUS);

    auto tb0 = Clock::now();
    mb.gaussian_blur(image, blurred, kernel, GAUSS_RADIUS);
    double tb = Sec(Clock::now() - tb0).count();

    std::cout << "  Tiempo: " << tb << " s\n";
    ImageIO::save_ppm("output/mandelbrot_8k_blurred.ppm", blurred, WIDTH, HEIGHT);

    // ------------------------------------------------------------------
    // Resumen
    // ------------------------------------------------------------------
    std::cout << "\n--------------------------------------------\n"
              << "  Tarea A : " << ta << " s\n"
              << "  Tarea B : " << tb << " s\n"
              << "  TOTAL   : " << (ta + tb) << " s\n"
              << "--------------------------------------------\n";

    return 0;
}
