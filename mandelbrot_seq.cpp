/*
 * mandelbrot_seq.cpp  –  Versión SECUENCIAL
 *
 * Tarea A: Genera imagen 8K del conjunto de Mandelbrot con coloreo suave.
 * Tarea B: Aplica un filtro de desenfoque Gaussiano 2D (radio = 20, núcleo 41×41).
 *
 * Compilar:
 *   g++ -O2 -std=c++17 -o mandelbrot_seq mandelbrot_seq.cpp
 *
 * Salida:
 *   mandelbrot_8k.ppm          (fractal sin filtro)
 *   mandelbrot_8k_blurred.ppm  (fractal con Gaussiano)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <algorithm>

// ---------------------------------------------------------------------------
// Parámetros globales
// ---------------------------------------------------------------------------
static constexpr int WIDTH       = 7680;
static constexpr int HEIGHT      = 4320;
static constexpr int MAX_ITER    = 1024;
static constexpr double X_MIN    = -2.5;
static constexpr double X_MAX    =  1.0;
static constexpr double Y_MIN    = -1.25;
static constexpr double Y_MAX    =  1.25;
static constexpr int GAUSS_RADIUS = 20;   // núcleo 41×41

// ---------------------------------------------------------------------------
// Tipos
// ---------------------------------------------------------------------------
struct Pixel { uint8_t r, g, b; };

// ---------------------------------------------------------------------------
// Paleta de color suave para Mandelbrot
// ---------------------------------------------------------------------------
static Pixel smooth_color(int iter, double zr, double zi) {
    if (iter == MAX_ITER) return {0, 0, 0};

    // Conteo de iteraciones suavizado (evita bandas duras)
    double log_zn = std::log(zr * zr + zi * zi) * 0.5;
    double nu     = std::log(log_zn / std::log(2.0)) / std::log(2.0);
    double t      = (iter + 1 - nu) / MAX_ITER;
    t = std::max(0.0, std::min(1.0, t));

    // Paleta "ultra fractal" (azul → cian → blanco → naranja → negro)
    auto lerp = [](double a, double b, double x) { return a + (b - a) * x; };

    double r, g, b;
    if (t < 0.16) {
        double s = t / 0.16;
        r = lerp(0,   32,  s);
        g = lerp(7,   107, s);
        b = lerp(100, 203, s);
    } else if (t < 0.42) {
        double s = (t - 0.16) / 0.26;
        r = lerp(32,  237, s);
        g = lerp(107, 255, s);
        b = lerp(203, 255, s);
    } else if (t < 0.6425) {
        double s = (t - 0.42) / 0.2225;
        r = lerp(237, 255, s);
        g = lerp(255, 170, s);
        b = lerp(255, 0,   s);
    } else if (t < 0.8575) {
        double s = (t - 0.6425) / 0.215;
        r = lerp(255, 0,   s);
        g = lerp(170, 2,   s);
        b = lerp(0,   0,   s);
    } else {
        double s = (t - 0.8575) / 0.1425;
        r = lerp(0,  0, s);
        g = lerp(2,  7, s);
        b = lerp(0, 100, s);
    }

    return { static_cast<uint8_t>(r),
             static_cast<uint8_t>(g),
             static_cast<uint8_t>(b) };
}

// ---------------------------------------------------------------------------
// TAREA A – Cálculo de un píxel del conjunto de Mandelbrot
// ---------------------------------------------------------------------------
static Pixel compute_mandelbrot(int px, int py) {
    double cr = X_MIN + (X_MAX - X_MIN) * px / (WIDTH  - 1);
    double ci = Y_MIN + (Y_MAX - Y_MIN) * py / (HEIGHT - 1);

    double zr = 0.0, zi = 0.0;
    int    iter = 0;

    while (iter < MAX_ITER && (zr * zr + zi * zi) <= 4.0) {
        double tmp = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = tmp;
        ++iter;
    }

    return smooth_color(iter, zr, zi);
}

// ---------------------------------------------------------------------------
// TAREA B – Generación del kernel Gaussiano 2D
// ---------------------------------------------------------------------------
static std::vector<double> make_gaussian_kernel(int radius) {
    const double sigma = radius / 3.0;
    const int    size  = 2 * radius + 1;
    std::vector<double> k(size * size);
    double sum = 0.0;

    for (int y = -radius; y <= radius; ++y)
        for (int x = -radius; x <= radius; ++x) {
            double v = std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
            k[(y + radius) * size + (x + radius)] = v;
            sum += v;
        }

    for (auto& v : k) v /= sum;   // normalizar
    return k;
}

// ---------------------------------------------------------------------------
// TAREA B – Convolución Gaussiana 2D (núcleo completo, NO separable)
// ---------------------------------------------------------------------------
static void gaussian_blur(const std::vector<Pixel>& src,
                           std::vector<Pixel>&       dst,
                           const std::vector<double>& kernel,
                           int radius)
{
    const int size = 2 * radius + 1;

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
// Guardar en formato PPM binario (P6)
// ---------------------------------------------------------------------------
static void save_ppm(const std::string& path,
                     const std::vector<Pixel>& img)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "No se pudo abrir: " << path << "\n"; return; }
    f << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data()),
            static_cast<std::streamsize>(img.size() * sizeof(Pixel)));
    std::cout << "  Guardado: " << path << "\n";
}

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

    std::vector<Pixel> image(WIDTH * HEIGHT);
    std::vector<Pixel> blurred(WIDTH * HEIGHT);

    // ------------------------------------------------------------------
    // TAREA A
    // ------------------------------------------------------------------
    std::cout << "[Tarea A] Generando fractal de Mandelbrot...\n";
    auto ta0 = Clock::now();

    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
            image[y * WIDTH + x] = compute_mandelbrot(x, y);

    double ta = Sec(Clock::now() - ta0).count();
    std::cout << "  Tiempo: " << ta << " s\n";
    save_ppm("mandelbrot_8k.ppm", image);

    // ------------------------------------------------------------------
    // TAREA B
    // ------------------------------------------------------------------
    std::cout << "\n[Tarea B] Aplicando desenfoque Gaussiano 2D...\n";
    auto kernel = make_gaussian_kernel(GAUSS_RADIUS);

    auto tb0 = Clock::now();
    gaussian_blur(image, blurred, kernel, GAUSS_RADIUS);
    double tb = Sec(Clock::now() - tb0).count();

    std::cout << "  Tiempo: " << tb << " s\n";
    save_ppm("mandelbrot_8k_blurred.ppm", blurred);

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
