CXX := g++
CXXFLAGS := -O2 -std=c++17
OMPFLAGS := -fopenmp
BIN_DIR := bin

SEQ_SRC := mandelbrot_seq.cpp
OMP_SRC := mandelbrot_omp.cpp
BENCH_SRC := mandelbrot_sched_bench.cpp
HIST_SRC := mandelbrot_histogram.cpp

SEQ_BIN := $(BIN_DIR)/mandelbrot_seq
OMP_BIN := $(BIN_DIR)/mandelbrot_omp
BENCH_BIN := $(BIN_DIR)/mandelbrot_sched_bench
HIST_BIN := $(BIN_DIR)/mandelbrot_histogram

.PHONY: all bench omp seq histogram clean

all: bench omp seq histogram

bench: $(BENCH_BIN)

omp: $(OMP_BIN)

seq: $(SEQ_BIN)

histogram: $(HIST_BIN)

$(SEQ_BIN): $(SEQ_SRC) Mandelbrot.hpp ImageIO.hpp ColorPallete.hpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(OMP_BIN): $(OMP_SRC) Mandelbrot.hpp ImageIO.hpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

$(BENCH_BIN): $(BENCH_SRC) Mandelbrot.hpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

$(HIST_BIN): $(HIST_SRC) ImageIO.hpp ColorPallete.hpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

clean:
	rm -f $(SEQ_BIN) $(OMP_BIN) $(BENCH_BIN) $(HIST_BIN)
