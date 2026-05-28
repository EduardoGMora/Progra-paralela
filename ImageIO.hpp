#ifndef IMAGE_IO_HPP
#define IMAGE_IO_HPP

#include "ColorPallete.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

// Utilidades de E/S de imágenes. No se puede instanciar.
class ImageIO {
public:
    ImageIO() = delete;

    // Guarda una imagen en formato PPM binario (P6).
    static void save_ppm(const std::string& path,
                         const std::vector<Pixel>& img,
                         int width, int height) {
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "No se pudo abrir: " << path << "\n"; return; }
        f << "P6\n" << width << " " << height << "\n255\n";
        f.write(reinterpret_cast<const char*>(img.data()),
                static_cast<std::streamsize>(img.size() * sizeof(Pixel)));
        std::cout << "  Guardado: " << path << "\n";
    }
    // Carga una imagen en formato PPM binario (P6). Devuelve false si falla.
    static bool load_ppm(const std::string& path,
                         std::vector<Pixel>& img,
                         int& width, int& height) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "No se pudo abrir: " << path << "\n"; return false; }
        std::string magic;
        int maxval;
        f >> magic >> width >> height >> maxval;
        if (magic != "P6" || maxval != 255) {
            std::cerr << "Formato PPM no soportado\n"; return false;
        }
        f.get(); // consume el '\n' tras el encabezado
        img.resize(width * height);
        f.read(reinterpret_cast<char*>(img.data()),
               static_cast<std::streamsize>(img.size() * sizeof(Pixel)));
        return f.good();
    }
};

#endif // IMAGE_IO_HPP
