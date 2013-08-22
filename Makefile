OPT = -O2

# CPU stuff
CC = g++-4.7
CFLAGS = -g $(OPT) -Wall -std=c++11
SRCS = lodepng.cpp

# GPU stuff
CUDA_PATH     ?= /usr/local/cuda
CUDA_BIN_PATH ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH ?= $(CUDA_PATH)/lib$(shell getconf LONG_BIT)
CUDA_INC_PATH ?= $(CUDA_PATH)/include

CUDA = $(CUDA_BIN_PATH)/nvcc
CUDA_FLAGS = $(OPT) -Wall -arch=sm_35
CUDA_SRCS = hbp.cu stereo.cu

# Linker settings
LDFLAGS = -I$(CUDA_INC_PATH) -L$(CUDA_LIB_PATH) -lcudart

TARGETS = stereo

OBJS = $(SRCS:.cpp=.o) $(CUDA_SRCS:.cu=.cuo)

all: $(TARGETS)

.PHONEY: test clean

$(TARGETS): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.cuo: %.cu
	$(CUDA) $(CUDA_FLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

test: stereo
	mkdir -p out
	./stereo 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png

pairs: stereo
	mkdir -p out
	./stereo 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png
	./stereo 20 data/venus/imL.png data/venus/imR.png out/venus.png
	./stereo 60 data/cones/imL.png data/cones/imR.png out/cones.png
	./stereo 60 data/teddy/imL.png data/teddy/imR.png out/teddy.png

clean:
	rm *.o
	rm *.cuo
	rm $(TARGETS)
