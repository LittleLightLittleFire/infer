# Build options
LIBRARY = libinfer.a
GPU_SUPPORT = true
DEBUG = true

OPT = -O2

# Compiler set up for my convenience
# you should edit this section and set your compiler of choice
ifeq ($(shell hash clang++ 2> /dev/null; echo $$?), 0)
    CC = clang++
else
    CC = g++-4.7
endif

CFLAGS = $(OPT) -Wextra -std=c++11
LDFLAGS =

# CUDA set up
CUDA_PATH     ?= /usr/local/cuda
CUDA_BIN_PATH ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH ?= $(CUDA_PATH)/lib$(shell getconf LONG_BIT)
CUDA_INC_PATH ?= $(CUDA_PATH)/include

CUDA = $(CUDA_BIN_PATH)/nvcc
CUDA_CFLAGS = $(OPT) -arch=sm_35

ifdef DEBUG
    CFLAGS += -g
    CUDA_CFLAGS += -g -G
endif

ifdef GPU_SUPPORT
    LDFLAGS += -I$(CUDA_INC_PATH) -L$(CUDA_LIB_PATH) -lcudart
endif

