include ../config.mk

CFLAGS += -I../include

ifdef GPU_SUPPORT
    CFLAGS += -DGPU_SUPPORT
endif

# the order the linker searches for libraries is important, we specify where
# libinfer's dependencies after we link it
LDFLAGS := -L../ -linfer $(LDFLAGS)

.PHONY: stereo test pairs memcheck valgrind clean

TARGETS = driver

all: $(TARGETS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

driver: driver.o lodepng.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: driver
	mkdir -p out
	./$< bp stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png

pairs: driver
	mkdir -p out
	./$< bp stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png
	./$< bp stereo 20 8 data/venus/imL.png data/venus/imR.png out/venus.png
	./$< bp stereo 60 4 data/cones/imL.png data/cones/imR.png out/cones.png
	./$< bp stereo 60 4 data/teddy/imL.png data/teddy/imR.png out/teddy.png

memcheck: driver
	mkdir -p out
	$(CUDA_BIN_PATH)/cuda-memcheck ./$< gpu_bp stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png

valgrind: driver
	mkdir -p out
	valgrind ./$< bp stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png

clean:
	-rm *.o
	-rm $(TARGETS)
	-rm -r out
