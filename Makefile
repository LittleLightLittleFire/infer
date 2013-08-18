CC = clang++
CFLAGS = -g -O2 -Wall -std=c++11
LDFLAGS =
SRCS = stereo.cpp lodepng.cpp mst.cpp
TARGETS = stereo

OBJS = $(SRCS:.cpp=.o)

all: $(TARGETS)

.PHONEY: test clean

$(TARGETS): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

test: stereo
	./stereo data/tsukuba/imL.png data/tsukuba/imR.png output.png

pairs: stereo
	./stereo data/tsukuba/imL.png data/tsukuba/imR.png tsukuba.png
	./stereo data/cones/imL.png data/cones/imR.png cones.png
	./stereo data/teddy/imL.png data/teddy/imR.png teddy.png
	./stereo data/venus/imL.png data/venus/imR.png teddy.png

clean:
	rm *.o
	rm $(TARGETS)
