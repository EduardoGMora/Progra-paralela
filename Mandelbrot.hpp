#ifndef MANDELBROT_HPP
#define MANDELBROT_HPP

#include "ColorPallete.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// Clase reutilizable para el cálculo del fractal de Mandelbrot.
// Es thread-safe: todos sus miembros son const y compute() es puro.
class Mandelbrot {
public:
    const int    width, height, max_iter;
    const double x_min, x_max, y_min, y_max;

    constexpr Mandelbrot(int w, int h, int mi,
                         double xn = -2.5, double xx = 1.0,
                         double yn = -1.25, double yx = 1.25)
        : width(w), height(h), max_iter(mi),
          x_min(xn), x_max(xx), y_min(yn), y_max(yx) {}

    // Calcula el color de un píxel (px, py). Seguro para uso en hilos paralelos.
    inline Pixel compute(int px, int py) const {
        double cr = x_min + (x_max - x_min) * px / (width  - 1);
        double ci = y_min + (y_max - y_min) * py / (height - 1);
        double zr = 0.0, zi = 0.0;
        int iter = 0;
        while (iter < max_iter && (zr*zr + zi*zi) <= 4.0) {
            double tmp = zr*zr - zi*zi + cr;
            zi = 2.0*zr*zi + ci;
            zr = tmp;
            ++iter;
        }
        return ColorPalette::smooth_color(iter, zr, zi, max_iter);
    }

    // Genera un kernel gaussiano 2D normalizado de tamaño (2*radius+1)^2.
    static std::vector<double> make_gaussian_kernel(int radius) {
        const double sigma = radius / 3.0;
        const int    size  = 2 * radius + 1;
        std::vector<double> k(size * size);
        double sum = 0.0;
        for (int y = -radius; y <= radius; ++y)
            for (int x = -radius; x <= radius; ++x) {
                double v = std::exp(-(x*x + y*y) / (2.0 * sigma * sigma));
                k[(y + radius) * size + (x + radius)] = v;
                sum += v;
            }
        for (auto& v : k) v /= sum;
        return k;
    }

    // Convolución gaussiana 2D secuencial. Usa width/height de la instancia.
    void gaussian_blur(const std::vector<Pixel>& src,
                       std::vector<Pixel>&       dst,
                       const std::vector<double>& kernel,
                       int radius) const {
        const int size = 2 * radius + 1;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double r = 0.0, g = 0.0, b = 0.0;
                for (int ky = -radius; ky <= radius; ++ky) {
                    int ny = std::clamp(y + ky, 0, height - 1);
                    for (int kx = -radius; kx <= radius; ++kx) {
                        int nx = std::clamp(x + kx, 0, width - 1);
                        double w = kernel[(ky + radius) * size + (kx + radius)];
                        const Pixel& p = src[ny * width + nx];
                        r += w * p.r;
                        g += w * p.g;
                        b += w * p.b;
                    }
                }
                dst[y * width + x] = {
                    static_cast<uint8_t>(std::clamp(r, 0.0, 255.0)),
                    static_cast<uint8_t>(std::clamp(g, 0.0, 255.0)),
                    static_cast<uint8_t>(std::clamp(b, 0.0, 255.0))
                };
            }
        }
    }
};

#endif // MANDELBROT_HPP
