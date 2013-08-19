CC = clang++
CFLAGS = -g -O2 -Wall -std=c++11
LDFLAGS =
SRCS = stereo.cpp lodepng.cpp mst.cpp bp.cpp
TARGETS = stereo

OBJS = $(SRCS:.cpp=.o)

all: $(TARGETS)

.PHONEY: test clean

$(TARGETS): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

test: stereo
	./stereo 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png

pairs: stereo
	./stereo 16 data/tsukuba/imL.png data/tsukuba/imR.png out/tsukuba.png
	./stereo 20 data/venus/imL.png data/venus/imR.png out/venus.png
	./stereo 60 data/cones/imL.png data/cones/imR.png out/cones.png
	./stereo 60 data/teddy/imL.png data/teddy/imR.png out/teddy.png

clean:
	-rm *.o
	-rm $(TARGETS)
	-rm out/*
