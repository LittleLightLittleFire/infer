#include <iterator>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <cstdio>

#include <cuda.h>

#include "util.h"
#include "hbp.h"

#include "lodepng.h"

namespace {
    typedef unsigned int uint;
    typedef unsigned char uchar;

    // constants initalisation
    const float linear_scaling = 0.075;
    const float data_disc = 1.7;
    const float data_trunc = 15;

    __device__ float colour_to_grey(const uchar *rgb) {
        return 0.2989 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
    }

    __global__ void compute_unary(const uint width, const uint height, const uint labels, const uchar *left, const uchar *right, float *out) {
        const uint x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const uint y = (blockIdx.y * blockDim.y) + threadIdx.y;

        // do a bounds check
        if (x >= width || y >= height) {
            return;
        }

        const uint index = x + width * y;

        // initalise the side with 0s
        if (x < labels - 1) {
            for (uint p = 0; p < labels; ++p) {
                out[index * labels + p] = 0;
            }
            return;
        }

        for (uint p = 0; p < labels; ++p) {
            const float left_colour = colour_to_grey(left + index * 4);
            const float right_colour = colour_to_grey(right + (index - p) * 4);

            out[index * labels + p] = linear_scaling * fminf(abs(left_colour - right_colour), data_trunc);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cout << "usage ./stero [labels] [left.png] [right.png] [output.png]" << std::endl;
        return 1;
    }

    const uint labels = atoi(argv[1]);
    const char *left_name = argv[2];
    const char *right_name = argv[3];
    const char *output_name = argv[4];

    std::vector<uchar> left, right;
    uint width, height;

    { // image loading
        if (lodepng::decode(left, width, height, left_name) || lodepng::decode(right, width, height, right_name)) {
            std::cout << "error loading images" << std::endl;
            return 1;
        }
    }

    // underlying model is of a grid
    const uint nodes = width * height;

    // a 2d array of [index, label] represented as a flat array
    std::vector<float> unary_psi(nodes * labels);

    // load it into GPU memory
    float *dev_unary_psi;
    uchar *dev_left, *dev_right;

    cudaMalloc(&dev_unary_psi, nodes * labels * sizeof(float));
    cudaMalloc(&dev_left, left.size() * sizeof(uchar));
    cudaMalloc(&dev_right, right.size() * sizeof(uchar));

    cudaMemcpy(dev_left, &left[0], left.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_right, &right[0], right.size(), cudaMemcpyHostToDevice);

    const uint tile_size = 16;
    dim3 block_size(tile_size, tile_size);
    dim3 grid_size(ceil(static_cast<float>(width) / block_size.x), ceil(static_cast<float>(height) / block_size.y));

    compute_unary<<<grid_size, block_size>>>(width, height, labels, dev_left, dev_right, dev_unary_psi);

    std::vector<uchar> result = decode_hbp(labels, 5, 5, width, height, dev_unary_psi, data_disc);

    // clean up
    cudaFree(dev_unary_psi);
    cudaFree(dev_left);
    cudaFree(dev_right);

    // convert the results into an image
    std::vector<uchar> image(result.size() * 4);
    for (uint i = 0; i < result.size(); ++i) {
        const float val = result[i] * (256.0f / labels);
        image[i * 4] = image[i * 4 + 1] = image[i * 4 + 2] = val;
        image[i * 4 + 3] = 255; // alpha channel
    }

    if (lodepng::encode(output_name, image, width, height)) {
        std::cout << "error writing image" << std::endl;
        return 2;
    }

    return 0;
}
