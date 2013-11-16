# Build options
LIBRARY = libinfer.a
GPU_SUPPORT = true

# Compiler set up
CC = clang++
CFLAGS = -g -O2 -Wextra -std=c++11
LDFLAGS =

# CUDA set up
CUDA_PATH     ?= /usr/local/cuda
CUDA_BIN_PATH ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH ?= $(CUDA_PATH)/lib$(shell getconf LONG_BIT)
CUDA_INC_PATH ?= $(CUDA_PATH)/include

CUDA = $(CUDA_BIN_PATH)/nvcc
CUDA_FLAGS = $(OPT) -g -G -arch=sm_35

ifdef GPU_SUPPORT
	LD_FLAGS += -I$(CUDA_INC_PATH) -L$(CUDA_LIB_PATH) -lcudart
endif

