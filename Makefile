# Makefile for CUDA program

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -O3

# Target executable name
TARGET = vector_sqrt_compare

# Source files
SRC = vector_sqrt_compare.cu

# Default target
all: $(TARGET)

# Rule for building the target executable
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

# Rule for running the program
run: $(TARGET)
	./$(TARGET)

# Clean up
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all run clean

