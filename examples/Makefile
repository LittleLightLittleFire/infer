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

driver: driver.o lodepng.o stereo.o iseg.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: driver
	mkdir -p out
	#./$< bp -v stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png
	#./$< gpu_bp -v stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png
	./$< hbp -v stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba_hbp.png
	#./$< gpu_hbp -v stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba_hbp.png
	#./$< trhbp -v stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba_trhbp.png
	#./$< trhbp -v -a stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba_trhbp_a.png
	#./$< hbp -v iseg data/leafpav/leafpav.png data/leafpav/leafpav_anno.png out/leafpav.png
	#./$< bp -v -r 20 iseg data/leafpav/leafpav.png data/leafpav/leafpav_anno.png out/leafpav.png
	#./$< qp iseg data/postjp/postjp.png data/postjp/postjp_anno.png out/postjp.png
	#./$< qp iseg data/palovigna/palovigna.png data/palovigna/palovigna_anno.png out/palovigna.png
	#./$< qp iseg data/tendrils/tendrils.png data/tendrils/tendrils_anno.png out/tendrils.png

pairs: driver
	mkdir -p out
	./$< bp stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png
	./$< bp stereo 20 8 data/venus/imL.png data/venus/imR.png out/venus.png
	./$< bp stereo 60 4 data/cones/imL.png data/cones/imR.png out/cones.png
	./$< bp stereo 60 4 data/teddy/imL.png data/teddy/imR.png out/teddy.png

memcheck: driver
	mkdir -p out
	$(CUDA_BIN_PATH)/cuda-memcheck ./$< gpu_hbp -v stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png

valgrind: driver
	mkdir -p out
	valgrind ./$< bp stereo 16 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png
	valgrind ./$< bp iseg data/leafpav/leafpav.png data/leafpav/leafpav_anno.png out/leafpav.png

clean:
	-rm *.o
	-rm $(TARGETS)
	-rm -r out
