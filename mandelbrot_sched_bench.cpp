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
#include <cmath>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <string>
#include <limits>
#include <omp.h>

static constexpr int    W        = 3840;
static constexpr int    H        = 2160;
static constexpr int    MAX_ITER = 512;
static constexpr double X_MIN    = -2.5;
static constexpr double X_MAX    =  1.0;
static constexpr double Y_MIN    = -1.25;
static constexpr double Y_MAX    =  1.25;
static constexpr int    RUNS     = 3;

struct Pixel { uint8_t r, g, b; };

static Pixel smooth_color(int iter, double zr, double zi) {
    if (iter == MAX_ITER) return {0, 0, 0};
    double log_zn = std::log(zr * zr + zi * zi) * 0.5;
    double nu     = std::log(log_zn / std::log(2.0)) / std::log(2.0);
    double t      = std::max(0.0, std::min(1.0, (iter + 1 - nu) / MAX_ITER));
    auto lerp = [](double a, double b, double x){ return a + (b - a) * x; };
    double r, g, b;
    if      (t < 0.16)   { double s = t / 0.16;           r=lerp(0,32,s);  g=lerp(7,107,s);   b=lerp(100,203,s); }
    else if (t < 0.42)   { double s = (t-0.16)/0.26;      r=lerp(32,237,s);g=lerp(107,255,s); b=lerp(203,255,s); }
    else if (t < 0.6425) { double s = (t-0.42)/0.2225;    r=lerp(237,255,s);g=lerp(255,170,s);b=lerp(255,0,s);   }
    else if (t < 0.8575) { double s = (t-0.6425)/0.215;   r=lerp(255,0,s); g=lerp(170,2,s);   b=lerp(0,0,s);     }
    else                 { double s = (t-0.8575)/0.1425;   r=lerp(0,0,s);   g=lerp(2,7,s);     b=lerp(0,100,s);   }
    return { (uint8_t)r, (uint8_t)g, (uint8_t)b };
}

static inline Pixel compute_mandelbrot(int px, int py) {
    double cr = X_MIN + (X_MAX - X_MIN) * px / (W - 1);
    double ci = Y_MIN + (Y_MAX - Y_MIN) * py / (H - 1);
    double zr = 0.0, zi = 0.0;
    int iter = 0;
    while (iter < MAX_ITER && (zr*zr + zi*zi) <= 4.0) {
        double tmp = zr*zr - zi*zi + cr;
        zi = 2.0*zr*zi + ci;
        zr = tmp;
        ++iter;
    }
    return smooth_color(iter, zr, zi);
}

// ---------------------------------------------------------------------------
// Kernels con planificador fijo en tiempo de compilación
// ---------------------------------------------------------------------------

static void kern_static(std::vector<Pixel>& img) {
    #pragma omp parallel for schedule(static) default(none) shared(img)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            img[y*W+x] = compute_mandelbrot(x, y);
}

// --- dynamic ---
static void kern_dyn(std::vector<Pixel>& img, int chunk=1) {
    #pragma omp parallel for schedule(dynamic,chunk) default(none) shared(img) firstprivate(chunk)
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) img[y*W+x] = compute_mandelbrot(x,y);
}

// --- guided ---
static void kern_guid_def(std::vector<Pixel>& img, int /*chunk*/) {
    #pragma omp parallel for schedule(guided) default(none) shared(img)
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) img[y*W+x] = compute_mandelbrot(x,y);
}
static void kern_guid(std::vector<Pixel>& img, int chunk=1) {
    #pragma omp parallel for schedule(guided,chunk) default(none) shared(img) firstprivate(chunk)
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) img[y*W+x] = compute_mandelbrot(x,y);
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
