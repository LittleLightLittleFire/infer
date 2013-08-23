#include <limits>
#include <vector>
#include <iostream>
#include <cstdio>

#include <cuda.h>
#include <math_constants.h>

namespace {
    typedef unsigned char uchar;
    typedef unsigned uint;

    __device__ float *ndx(const uint labels, const uint width, float *dir, const uint x, const uint y) {
        return labels * (x + y * width) + dir;
    }

    __device__ const float *cndx(const uint labels, const uint width, const float *dir, const uint x, const uint y) {
        return labels * (x + y * width) + dir;
    }

    /** generate the next layer's potentials */
    __global__ void fill_next_layer_pot(const uint labels, const uint width, const uint height, const uint max_width, const uint max_height, const float *pot, float *out) {
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;

        // bounds check
        if (x >= width || y >= height) {
            return;
        }

        // collapse the potential in a 2x2 area
        float *target = ndx(labels, width, out, x, y);
        const float *top_left = cndx(labels, max_width, pot, 2 * x, 2 * y);;

        for (uint i = 0; i < labels; ++i) {
            target[i] = top_left[i];
        }

        if (2 * x + 1 < max_width) {
            const float *top_right = cndx(labels, max_width, pot, 2 * x + 1, 2 * y);;
            for (uint i = 0; i < labels; ++i) {
                target[i] += top_right[i];
            }
        }

        if (2 * (y + 1) < max_height) {
            const float *bottom_left = cndx(labels, max_width, pot, 2 * x, 2 * (y + 1));
            for (uint i = 0; i < labels; ++i) {
                target[i] += bottom_left[i];
            }
        }

        if (2 * x + 1 < max_width && 2 * (y + 1) < max_height) {
            const float *bottom_right = cndx(labels, max_width, pot, 2 * x + 1, 2 * (y + 1));
            for (uint i = 0; i < labels; ++i) {
                target[i] += bottom_right[i];
            }
        }
    }

    /** max product send message */
    __device__ void send_msg_map(const uint labels, const float disc_trunc, const float *m1, const float *m2, const float *m3, const float *pot, float *out) {
        float curr_min = CUDART_MAX_NORMAL_F;

        // add all the incoming messages together
        for (uint i = 0; i < labels; ++i) {
            out[i] = m1[i] + m2[i] + m3[i] + pot[i];
            curr_min = fminf(curr_min, out[i]);
        }

        // do the O(n) trick
        for (uint i = 1; i < labels; ++i) {
            out[i] = fminf(out[i-1] + 1.0, out[i]);
        }

        for (int i = labels - 2; i >= 0; --i) {
            out[i] = fminf(out[i+1] + 1.0, out[i]);
        }

        // truncate
        for (uint i = 0; i < labels; ++i) {
            out[i] = fminf(curr_min + disc_trunc, out[i]);
        }

        // normalise
        float sum = 0;
        for (uint i = 0; i < labels; ++i) {
            sum += out[i];
        }

        sum /= static_cast<float>(labels);
        for (uint i = 0; i < labels; ++i) {
            out[i] -= sum;
        }
    }

    /** loopy belief propagation */
    __global__ void bp(const uint lbl, const uint w, const uint h, const float disc_trunc, const uint i, const float *pot, float *u, float *d, float *l, float *r) {
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;

        // bounds check
        if (x < 1 || y < 1 || x >= w - 1|| y >= h - 1) {
            return;
        }

        // check if this thread is active for this iteration
        if ((x + y + i) % 2 == 0) {
            send_msg_map(lbl, disc_trunc, cndx(lbl, w, u, x, y+1), cndx(lbl, w, l , x+1, y), cndx(lbl, w, r, x-1, y), cndx(lbl, w, pot, x, y), ndx(lbl, w, u, x, y));
            send_msg_map(lbl, disc_trunc, cndx(lbl, w, d, x, y-1), cndx(lbl, w, l , x+1, y), cndx(lbl, w, r, x-1, y), cndx(lbl, w, pot, x, y), ndx(lbl, w, d, x, y));
            send_msg_map(lbl, disc_trunc, cndx(lbl, w, u, x, y+1), cndx(lbl, w, d, x, y-1),  cndx(lbl, w, r, x-1, y), cndx(lbl, w, pot, x, y), ndx(lbl, w, r, x, y));
            send_msg_map(lbl, disc_trunc, cndx(lbl, w, u, x, y+1), cndx(lbl, w, d, x, y-1),  cndx(lbl, w, l, x+1, y), cndx(lbl, w, pot, x, y), ndx(lbl, w, l, x, y));
        }
    }

    /** initalise messages using the messages from the layer below */
    __global__ void prime(const uint lbl, const uint w, const uint h, const uint prev_w, const float *prev_msg, float *out) {
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;

        // boundary check
        if (x >= w || y >= h) {
            return;
        }

        // initaise to the last layer's (x/2, y/2)
        float *target = ndx(lbl, w, out, x, y);
        const float *source = cndx(lbl, prev_w, prev_msg, x / 2, y / 2);

        for (uint i = 0; i < lbl; ++i) {
            target[i] = source[i];
        }
    }

    __global__ void get_results(const uint lbl, const uint w, const uint h, const float *u, const float *d, const float *l, const float *r, const float *pot, uchar *out) {
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;

        // boundary check
        if (x < 1 || y < 1 || x >= w - 1|| y >= h - 1) {
            return;
        }

        uint min_label = 0;
        float min_value = CUDART_MAX_NORMAL_F;

        const float *ut = cndx(lbl, w, u, x, y+1);
        const float *dt = cndx(lbl, w, d, x, y-1);
        const float *lt = cndx(lbl, w, l, x+1, y);
        const float *rt = cndx(lbl, w, r, x-1, y);
        const float *pott = cndx(lbl, w, pot, x, y);

        for (uint i = 0; i < lbl; ++i) {
            const float val = ut[i] + dt[i] + lt[i] + rt[i] + pott[i];
            if (val < min_value) {
                min_label = i;
                min_value = val;
            }
        }

        out[x + y * w] = min_label;
    }
}

std::vector<uchar> decode_hbp(const uchar labels, const uint layers, const uint max_iter, const uint width, const uint height, const std::vector<float> &pot, const float disc_trunc) {
    const uint tile_size = 16;
    dim3 block(tile_size, tile_size);

    // move the potentials to device memory
    float *dev_pot_initial;
    cudaMalloc(&dev_pot_initial, pot.size() * sizeof(float));
    cudaMemcpy(dev_pot_initial, &pot[0], pot.size() * sizeof(float), cudaMemcpyHostToDevice);

    // pointers for the layers
    std::vector<float2> layer_sizes;
    std::vector<float *> dev_pot;

    // messages on the current layer and on the one below
    float *dev_u, *dev_d, *dev_l, *dev_r;
    float *dev_pu, *dev_pd, *dev_pl, *dev_pr;

    // inital set up
    dev_pot.push_back(dev_pot_initial);
    layer_sizes.push_back(make_float2(width, height));

    // create potentials for all layers
    for (uint i = 1; i < layers; ++i) {
        // calculate size of this layer
        const uint layer_width = (layer_sizes[i-1].x + 1) / 2, layer_height = (layer_sizes[i-1].y + 1) / 2;
        layer_sizes.push_back(make_float2(layer_width, layer_height));

        // memory for the potentials
        float *dev_layer_pot;
        cudaMalloc(&dev_layer_pot, labels * layer_width * layer_height * sizeof(float));
        dev_pot.push_back(dev_layer_pot);

        // call the kernel to create the potential
        dim3 grid((layer_width + tile_size - 1) / tile_size, (layer_height + tile_size - 1) / tile_size);
        fill_next_layer_pot<<<grid, block>>>(labels, layer_width, layer_height, layer_sizes[i-1].x, layer_sizes[i-1].y, dev_pot[i-1], dev_layer_pot);
    }

    { // initalise memory for the messages
        const uint top_size = labels * width * height * sizeof(float);
        cudaMalloc(&dev_u, top_size);
        cudaMalloc(&dev_d, top_size);
        cudaMalloc(&dev_l, top_size);
        cudaMalloc(&dev_r, top_size);

        cudaMalloc(&dev_pu, top_size);
        cudaMalloc(&dev_pd, top_size);
        cudaMalloc(&dev_pl, top_size);
        cudaMalloc(&dev_pr, top_size);

        const uint elems = labels * layer_sizes.back().x * layer_sizes.back().y;
        const uint size = elems * sizeof(float);

        cudaMemset(dev_u, 0, size);
        cudaMemset(dev_d, 0, size);
        cudaMemset(dev_l, 0, size);
        cudaMemset(dev_r, 0, size);
    }

    { // run an initial round of BP on the bottom layer
        dim3 grid((layer_sizes.back().x + tile_size - 1) / tile_size, (layer_sizes.back().y + tile_size - 1) / tile_size);

        for (uint i = 0; i < max_iter; ++i) {
            bp<<<grid, block>>>(labels, layer_sizes.back().x, layer_sizes.back().y, disc_trunc, i, dev_pot.back(), dev_u, dev_d, dev_l, dev_r);
        }

        std::swap(dev_u, dev_pu);
        std::swap(dev_d, dev_pd);
        std::swap(dev_l, dev_pl);
        std::swap(dev_r, dev_pr);
    }

    // create messages using the messages on the layer below
    for (int i = layers - 2; i >= 0; --i) {
        dim3 grid((layer_sizes[i].x + tile_size - 1) / tile_size, (layer_sizes[i].y + tile_size - 1) / tile_size);
        prime<<<grid, block>>>(labels, layer_sizes[i].x, layer_sizes[i].y, layer_sizes[i+1].x, dev_pu, dev_u);
        prime<<<grid, block>>>(labels, layer_sizes[i].x, layer_sizes[i].y, layer_sizes[i+1].x, dev_pd, dev_d);
        prime<<<grid, block>>>(labels, layer_sizes[i].x, layer_sizes[i].y, layer_sizes[i+1].x, dev_pl, dev_l);
        prime<<<grid, block>>>(labels, layer_sizes[i].x, layer_sizes[i].y, layer_sizes[i+1].x, dev_pr, dev_r);

        // run the bp for this layer
        for (uint j = 0; j < max_iter; ++j) {
            bp<<<grid, block>>>(labels, layer_sizes[i].x, layer_sizes[i].y, disc_trunc, j, dev_pot[i], dev_u, dev_d, dev_l, dev_r);
        }

        std::swap(dev_u, dev_pu);
        std::swap(dev_d, dev_pd);
        std::swap(dev_l, dev_pl);
        std::swap(dev_r, dev_pr);
    }

    std::vector<uchar> results(width * height);
    { // collect results
        uchar *dev_out;
        cudaMalloc(&dev_out, width * height * sizeof(uchar));

        dim3 grid((width + tile_size - 1) / tile_size, (height + tile_size - 1) / tile_size);
        get_results<<<grid, block>>>(labels, width, height, dev_pu, dev_pd, dev_pl, dev_pr, dev_pot_initial, dev_out);
        cudaMemcpy(&results[0], dev_out, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);
        cudaFree(dev_out);
    }

    { // clean up
        for (uint i = 0; i < dev_pot.size(); ++i) {
            cudaFree(dev_pot[i]);
        }

        cudaFree(dev_u);
        cudaFree(dev_d);
        cudaFree(dev_l);
        cudaFree(dev_r);

        cudaFree(dev_pu);
        cudaFree(dev_pd);
        cudaFree(dev_pl);
        cudaFree(dev_pr);
    }

    return results;
}
