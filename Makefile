CXX := g++
CXXFLAGS    := -O2 -std=c++17
CXXFLAGS_O3 := -O3 -std=c++17 -march=native
OMPFLAGS    := -fopenmp
BIN_DIR     := bin

SEQ_SRC    := mandelbrot_seq.cpp
OMP_SRC    := mandelbrot_omp.cpp
BENCH_SRC  := mandelbrot_sched_bench.cpp
HIST_SRC   := mandelbrot_histogram.cpp
SPMD_SRC   := mandelbrot_spmd.cpp
SCALING_SRC:= mandelbrot_scaling_bench.cpp

SEQ_BIN    := $(BIN_DIR)/mandelbrot_seq
OMP_BIN    := $(BIN_DIR)/mandelbrot_omp
BENCH_BIN  := $(BIN_DIR)/mandelbrot_sched_bench
HIST_BIN   := $(BIN_DIR)/mandelbrot_histogram
SPMD_BIN   := $(BIN_DIR)/mandelbrot_spmd
SCALING_BIN:= $(BIN_DIR)/mandelbrot_scaling_bench

.PHONY: all bench omp seq histogram spmd scaling plots clean

all: seq omp bench histogram spmd scaling

bench: $(BENCH_BIN)

omp: $(OMP_BIN)

seq: $(SEQ_BIN)

histogram: $(HIST_BIN)

# -fopt-info-vec imprime durante la compilación los bucles que se vectorizaron.
# Buscar líneas con "vectorized" en la salida del compilador.
spmd: $(SPMD_BIN)

scaling: $(SCALING_BIN)

# Ejecuta el benchmark de escalabilidad y genera las gráficas.
plots: $(SCALING_BIN)
	./$(SCALING_BIN)
	python3 plot_scaling.py

$(SEQ_BIN): $(SEQ_SRC) Mandelbrot.hpp ImageIO.hpp ColorPallete.hpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(OMP_BIN): $(OMP_SRC) Mandelbrot.hpp ImageIO.hpp ColorPallete.hpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

$(BENCH_BIN): $(BENCH_SRC) Mandelbrot.hpp ColorPallete.hpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

$(HIST_BIN): $(HIST_SRC) ImageIO.hpp ColorPallete.hpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

$(SPMD_BIN): $(SPMD_SRC) Mandelbrot.hpp ColorPallete.hpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS_O3) $(OMPFLAGS) -fopt-info-vec -o $@ $<

$(SCALING_BIN): $(SCALING_SRC) Mandelbrot.hpp ColorPallete.hpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

clean:
	rm -f $(SEQ_BIN) $(OMP_BIN) $(BENCH_BIN) $(HIST_BIN) $(SPMD_BIN) $(SCALING_BIN)
