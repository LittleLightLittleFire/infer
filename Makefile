include config.mk

# Makefile for the static library and examples

CFLAGS += -Iinclude
CUDA_CFLAGS += -Iinclude

# CPU sources
SRCS = crf.cpp method.cpp bp.cpp qp.cpp trbp.cpp mst.cpp

# GPU sources
CUDA_SRCS = crf.cu method.cu bp.cu util.cu trbp.cu core.cu

LIBRARY_OBJS = $(SRCS:.cpp=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.cuo)

ifdef GPU_SUPPORT
	LIBRARY_OBJS += $(CUDA_OBJS)
endif

all: $(LIBRARY) examples/stereo

.PHONEY: test clean

$(LIBRARY): $(LIBRARY_OBJS)
	ar rcs $@ $^
	cd examples && make clean

%.o: src/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.cuo: src/cuda/%.cu
	$(CUDA) $(CUDA_CFLAGS) -c $< -o $@

docs:
	doxygen

examples/%: $(LIBRARY)
	cd examples && make $*

memcheck: $(LIBRARY)
	cd examples && make $@

valgrind: $(LIBRARY)
	cd examples && make $@

test: $(LIBRARY)
	cd examples && make test

clean:
	-rm *.o
	-rm *.cuo
	-rm $(LIBRARY)
	-cd examples && make clean
	-rm -r out/
	-rm -r docs/
