CC = clang++
CFLAGS = -Wall -std=c++11
LDFLAGS =
SRCS = stereo.cpp lodepng.cpp
TARGETS = stereo

OBJS = $(SRCS:.cpp=.o)

all: $(TARGETS)

.PHONEY: test clean

$(TARGETS): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

test: stereo
	./stereo data/left.png data/right.png

clean:
	rm *.o
	rm $(TARGETS)
