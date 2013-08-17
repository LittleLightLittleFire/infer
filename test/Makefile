CFLAGS = -g -O2 -std=c++11

all: mst

mst: mst.cpp ../mst.cpp ../lodepng.cpp
	clang++ $(CFLAGS) $^ -o $@

clean:
	rm mst
