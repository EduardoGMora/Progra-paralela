/*
 * mandelbrot_histogram.cpp  –  Tarea C: Sincronización y Falsos Compartimientos
 *
 * Lee output/mandelbrot_8k_blurred.ppm (generada por mandelbrot_omp) y
 * calcula el histograma de colores (R, G, B) de tres formas distintas:
 *
 *  1. atomic        – #pragma omp atomic por cada incremento de bin.
 *                     Alta contención en bins populares → lento.
 *
 *  2. local+critical– Cada hilo acumula en arrays privados (sin contención)
 *                     y los fusiona al global con una sola sección crítica.
 *                     O(N) sin sincronización + O(256) con ella → muy rápido.
 *
 *  3. False Sharing – Demo explícito del fenómeno:
 *                     SIN padding: N hilos en un arreglo compacto comparten
 *                       líneas de caché → coherencia innecesaria entre núcleos.
 *                     CON padding: cada casilla ocupa 64 B (una línea propia)
 *                       → sin interferencia → significativamente más rápido.
 *                     Se compila con -O0 para que los acumuladores vayan a
 *                     memoria en cada iteración y el efecto sea medible.
 *
 * Requisito:
 *   Ejecutar primero mandelbrot_omp para generar output/mandelbrot_8k_blurred.ppm
 *
 * Compilar:
 *   g++ -O2 -fopenmp -std=c++17 -o bin/mandelbrot_histogram mandelbrot_histogram.cpp
 *
 * Ejecutar con N hilos:
 *   OMP_NUM_THREADS=N ./bin/mandelbrot_histogram
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include "ColorPallete.hpp"
#include "ImageIO.hpp"

using Clock = std::chrono::high_resolution_clock;
using Sec   = std::chrono::duration<double>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void print_hist_peak(const int hr[256], const int hg[256], const int hb[256]) {
    auto peak = [](const int h[]) -> std::pair<int,int> {
        int idx = (int)(std::max_element(h, h + 256) - h);
        return {idx, h[idx]};
    };
    auto [pr, cr] = peak(hr);
    auto [pg, cg] = peak(hg);
    auto [pb, cb] = peak(hb);
    std::cout << "    Pico R=(" << pr << ", " << cr << " px)"
              << "  G=(" << pg << ", " << cg << " px)"
              << "  B=(" << pb << ", " << cb << " px)\n";
}

// ---------------------------------------------------------------------------
// Impl. 1 – #pragma omp atomic
//
// Problema: cada uno de los ~100M píxeles (8K) dispara 3 operaciones atómicas.
// Los bins populares (p. ej. r=0 del interior negro del fractal) concentran
// millones de actualizaciones concurrentes → los hilos se serializan en ese bin.
// ---------------------------------------------------------------------------
static double histogram_atomic(const std::vector<Pixel>& img,
                                int hr[256], int hg[256], int hb[256])
{
    std::fill(hr, hr + 256, 0);
    std::fill(hg, hg + 256, 0);
    std::fill(hb, hb + 256, 0);
    const int N = (int)img.size();

    auto t0 = Clock::now();
    #pragma omp parallel for schedule(static) default(none) shared(img, hr, hg, hb, N)
    for (int i = 0; i < N; ++i) {
        #pragma omp atomic
        hr[img[i].r]++;
        #pragma omp atomic
        hg[img[i].g]++;
        #pragma omp atomic
        hb[img[i].b]++;
    }
    return Sec(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Impl. 2 – variables locales por hilo + #pragma omp critical al final
//
// Cada hilo acumula en arrays de stack (lr, lg, lb) que caben en su L1.
// Sin contención durante el bucle principal.
// Al terminar, fusiona sus 256×3 enteros al global con una sección crítica
// de tamaño fijo O(256), no O(N) → overhead despreciable.
// ---------------------------------------------------------------------------
static double histogram_local(const std::vector<Pixel>& img,
                               int hr[256], int hg[256], int hb[256])
{
    std::fill(hr, hr + 256, 0);
    std::fill(hg, hg + 256, 0);
    std::fill(hb, hb + 256, 0);
    const int N = (int)img.size();

    auto t0 = Clock::now();
    #pragma omp parallel default(none) shared(img, hr, hg, hb, N)
    {
        int lr[256] = {}, lg[256] = {}, lb[256] = {};

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < N; ++i) {
            lr[img[i].r]++;
            lg[img[i].g]++;
            lb[img[i].b]++;
        }

        #pragma omp critical
        for (int c = 0; c < 256; ++c) {
            hr[c] += lr[c];
            hg[c] += lg[c];
            hb[c] += lb[c];
        }
    }
    return Sec(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Demo de False Sharing
//
// Cada hilo escribe exclusivamente en count[tid], pero si varios count[i]
// caen en la misma línea de caché (64 B), el protocolo MESI obliga al
// hardware a invalidar esa línea en todos los demás núcleos en cada escritura.
// Esto genera tráfico de coherencia innecesario aunque no haya dependencia
// lógica entre los hilos.
//
// SIN PADDING: long long count[12] → 96 B, cabe en ~2 líneas de caché.
//   count[0..7] comparten línea con count[0] → falsa dependencia hardware.
//
// CON PADDING: PaddedLL count[12] → cada elemento ocupa 64 B exactos.
//   Cada hilo tiene su propia línea → cero interferencia.
//
// Se compila con -O0 para impedir que el compilador promueva count[tid]
// a un registro (lo que enmascaría el fenómeno).
// ---------------------------------------------------------------------------

struct alignas(64) PaddedLL {
    long long v;
    char pad[64 - sizeof(long long)];
};
static_assert(sizeof(PaddedLL) == 64, "PaddedLL debe tener exactamente 64 bytes");

#pragma GCC push_options
#pragma GCC optimize("O0")

static double bench_false_sharing_bad(const std::vector<Pixel>& img) {
    const int NT = omp_get_max_threads();
    std::vector<long long> count(NT, 0LL);
    const int N = (int)img.size();

    auto t0 = Clock::now();
    #pragma omp parallel default(none) shared(img, count, N)
    {
        const int tid   = omp_get_thread_num();
        const int nthrs = omp_get_num_threads();
        const int chunk = N / nthrs;
        const int start = tid * chunk;
        const int end   = (tid == nthrs - 1) ? N : start + chunk;
        for (int i = start; i < end; ++i)
            count[tid] += img[i].r + img[i].g + img[i].b;
    }
    return Sec(Clock::now() - t0).count();
}

static double bench_false_sharing_good(const std::vector<Pixel>& img) {
    const int NT = omp_get_max_threads();
    std::vector<PaddedLL> count(NT);
    for (auto& p : count) p.v = 0;
    const int N = (int)img.size();

    auto t0 = Clock::now();
    #pragma omp parallel default(none) shared(img, count, N)
    {
        const int tid   = omp_get_thread_num();
        const int nthrs = omp_get_num_threads();
        const int chunk = N / nthrs;
        const int start = tid * chunk;
        const int end   = (tid == nthrs - 1) ? N : start + chunk;
        for (int i = start; i < end; ++i)
            count[tid].v += img[i].r + img[i].g + img[i].b;
    }
    return Sec(Clock::now() - t0).count();
}

#pragma GCC pop_options

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const int nthreads = omp_get_max_threads();
    const std::string path = "output/mandelbrot_8k_blurred.ppm";

    std::cout << "============================================\n"
              << "  Tarea C: Histograma + False Sharing  [OpenMP]\n"
              << "============================================\n"
              << "  Hilos OpenMP: " << nthreads << "\n"
              << "  Imagen      : " << path << "\n\n";

    std::vector<Pixel> img;
    int width = 0, height = 0;

    std::cout << "Cargando imagen...\n";
    if (!ImageIO::load_ppm(path, img, width, height)) {
        std::cerr << "Error: ejecuta primero ./bin/mandelbrot_omp\n";
        return 1;
    }
    std::cout << "  " << width << " x " << height
              << "  (" << img.size() << " píxeles)\n\n";

    int hr[256], hg[256], hb[256];

    // ---- Histograma con atomic ----
    std::cout << "[Histograma] atomic (3 operaciones atómicas por píxel):\n";
    double t_at = histogram_atomic(img, hr, hg, hb);
    std::cout << "  Tiempo: " << std::fixed << std::setprecision(4) << t_at << " s\n";
    print_hist_peak(hr, hg, hb);

    // ---- Histograma con variables locales + critical ----
    std::cout << "\n[Histograma] local + critical (arrays privados por hilo):\n";
    double t_lc = histogram_local(img, hr, hg, hb);
    std::cout << "  Tiempo: " << std::fixed << std::setprecision(4) << t_lc << " s"
              << "  (speedup " << std::fixed << std::setprecision(2)
              << (t_at / t_lc) << "x vs atomic)\n";
    print_hist_peak(hr, hg, hb);

    // ---- Demo False Sharing ----
    std::cout << "\n[False Sharing] acumulador por hilo, sin optimización del compilador:\n";
    double t_bad  = bench_false_sharing_bad(img);
    double t_good = bench_false_sharing_good(img);

    std::cout << "  sin padding (false sharing) : "
              << std::fixed << std::setprecision(4) << t_bad << " s\n"
              << "  con padding (sin f.sharing) : "
              << std::fixed << std::setprecision(4) << t_good << " s";
    if (t_good > 0.0)
        std::cout << "  (speedup " << std::fixed << std::setprecision(2)
                  << (t_bad / t_good) << "x)";
    std::cout << "\n";

    if (t_bad > t_good * 1.05)
        std::cout << "  → False sharing detectado: escrituras adyacentes invalidaron\n"
                  << "    líneas de caché compartidas entre hilos.\n";
    else
        std::cout << "  → Sin diferencia medible: el bus de memoria enmascara el fenómeno.\n"
                  << "    El false sharing existe a nivel de hardware igualmente.\n";

    // ---- Resumen ----
    std::cout << "\n--------------------------------------------\n"
              << "  Histograma atomic    : " << std::fixed << std::setprecision(4) << t_at  << " s\n"
              << "  Histograma local+crit: " << std::fixed << std::setprecision(4) << t_lc  << " s"
              << "  (" << std::fixed << std::setprecision(2) << (t_at/t_lc) << "x)\n"
              << "  False sharing bad    : " << std::fixed << std::setprecision(4) << t_bad  << " s\n"
              << "  False sharing good   : " << std::fixed << std::setprecision(4) << t_good << " s"
              << "  (" << std::fixed << std::setprecision(2) << (t_bad/t_good) << "x)\n"
              << "  Hilos                : " << nthreads << "\n"
              << "--------------------------------------------\n";

    return 0;
}
