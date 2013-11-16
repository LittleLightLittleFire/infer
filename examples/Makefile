include ../config.mk

CFLAGS += -I../include
LDFLAGS += -L../ -linfer

TARGETS = stereo

all: $(TARGETS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

stereo: lodepng.o stereo.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: stereo
	mkdir -p out
	./$< 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png

pairs: stereo
	mkdir -p out
	./$< 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png
	./$< 20 8 data/venus/imL.png data/venus/imR.png out/venus.png
	./$< 60 4 data/cones/imL.png data/cones/imR.png out/cones.png
	./$< 60 4 data/teddy/imL.png data/teddy/imR.png out/teddy.png

memcheck: stereo
	mkdir -p out
	$(CUDA_BIN_PATH)/cuda-memcheck ./$< 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png

clean:
	-rm *.o
	-rm $(TARGETS)
