#ifndef COLOR_PALETTE_HPP
#define COLOR_PALETTE_HPP

#include <cstdint>
#include <cmath>
#include <algorithm>

struct Pixel { uint8_t r, g, b; };

// Clase utilitaria de paleta de color para el fractal de Mandelbrot.
// Todos los métodos son estáticos; no se puede instanciar.
class ColorPalette {
public:
    ColorPalette() = delete;

    // Coloreo suave basado en el número de escape y el módulo final de z.
    static inline Pixel smooth_color(int iter, double zr, double zi, int max_iter) {
        if (iter == max_iter) return {0, 0, 0};
        double log_zn = std::log(zr * zr + zi * zi) * 0.5;
        double nu     = std::log(log_zn / std::log(2.0)) / std::log(2.0);
        double t      = std::max(0.0, std::min(1.0, (iter + 1 - nu) / max_iter));
        auto lerp = [](double a, double b, double x){ return a + (b - a) * x; };
        double r, g, b;
        if      (t < 0.16)   { double s = t / 0.16;          r=lerp(0,32,s);  g=lerp(7,107,s);   b=lerp(100,203,s); }
        else if (t < 0.42)   { double s = (t-0.16)/0.26;     r=lerp(32,237,s);g=lerp(107,255,s); b=lerp(203,255,s); }
        else if (t < 0.6425) { double s = (t-0.42)/0.2225;   r=lerp(237,255,s);g=lerp(255,170,s);b=lerp(255,0,s);   }
        else if (t < 0.8575) { double s = (t-0.6425)/0.215;  r=lerp(255,0,s); g=lerp(170,2,s);   b=lerp(0,0,s);     }
        else                 { double s = (t-0.8575)/0.1425; r=lerp(0,0,s);   g=lerp(2,7,s);     b=lerp(0,100,s);   }
        return { (uint8_t)r, (uint8_t)g, (uint8_t)b };
    }
};

#endif // COLOR_PALETTE_HPP

