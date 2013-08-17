CFLAGS = -g -std=c++11

all: mst

mst: mst.cpp ../mst.cpp
	clang++ $(CFLAGS) $< ../mst.cpp -o $@

clean:
	rm mst
