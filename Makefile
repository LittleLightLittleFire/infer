include config.mk

# Makefile for the static library and examples

# CPU sources
SRCS = crf.cpp method.cpp bp.cpp qp.cpp trbp.cpp mst.cpp

# GPU sources
CUDA_SRCS =

LIBRARY_OBJS = $(SRCS:.cpp=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.cuo)

ifdef GPU_SUPPORT
	LIBRARY_OBJS += $(CUDA_OBJS)
endif

all: $(LIBRARY)

.PHONEY: test clean

$(LIBRARY): $(LIBRARY_OBJS)
	ar rcs $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.cuo: %.cu
	$(CUDA) $(CUDA_FLAGS) -c $< -o $@

docs:
	doxygen

examples/%: examples/%.cpp $(LIBRARY)
	cd examples && make $*

test: $(LIBRARY)
	cd examples && make test

clean:
	-rm *.o
	-rm $(LIBRARY)
	-cd examples && make clean
	-rm -r out/
	-rm -r docs/
