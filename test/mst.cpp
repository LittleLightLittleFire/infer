#include <iostream>
#include <algorithm>

#include "../mst.h"
#include "../lodepng.h"
#include "../util.h"

int main() {
    typedef unsigned char uchar;
    typedef unsigned int uint;

    const uint width = 384, height = 388;
    const edge_indexer edx(width, height);

    std::vector<uint> tree = sample_edge_apparence(width, height, 1);
    std::vector<uchar> image(width * height * 4, 255);

    const indexer idx(width * 2, height * 2);

    for (uint x = 0; x < width; ++x) {
        for (uint y = 0; y < height; ++y) {
            const uint index = idx(x * 2, y * 2);
            image[index] = 0;

            if (tree[edx(x, y, move::DOWN)]) {
                image[idx(x * 2, y * 2 + 1)] = 80;
            }

            if (tree[edx(x, y, move::RIGHT)]) {
                image[idx(x * 2 + 1, y * 2)] = 80;
            }
        }
    }

    // scale it up to argb
    std::vector<uchar> argb;
    for (const uchar val : image) {
        for (uint i = 0; i < 3; ++i) {
            argb.push_back(val);
        }

        argb.push_back(255);
    }

    if (lodepng::encode("maze.png", argb, width * 2, height * 2)) {
        std::cout << "error writing image" << std::endl;
    }
}
