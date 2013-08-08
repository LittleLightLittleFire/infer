CFLAGS = -std=c++11
EXTRA = `pkg-config --libs --cflags opencv`

all: stero

.PHONEY: test

stero: stero.cpp
	g++ $(CFLAGS) $(EXTRA) $< -o $@

test: stero
	./stero data/left.png data/right.png
