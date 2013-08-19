#ifndef UTIL_H
#define UTIL_H

#include <vector>

enum class move { UP, DOWN, LEFT, RIGHT };

/** Makes indexing two dimensional arrays a bit easier */
struct indexer {
    const unsigned width_, height_, scale_;
    explicit indexer(const unsigned width, const unsigned height, const unsigned scale = 1) : width_(width), height_(height), scale_(scale) { }

    /** returns the index in a 1d array */
    unsigned operator()(const unsigned x, const unsigned y) const {
        return scale_ * (x + width_ * y);
    }

    /** returns the node index after applying the operation */
    unsigned operator()(const unsigned idx, const move m) const {
        switch (m) {
            case move::UP:    return scale_ * (idx - width_);
            case move::DOWN:  return scale_ * (idx + width_);
            case move::LEFT:  return scale_ * (idx - 1);
            case move::RIGHT: return scale_ * (idx + 1);
        }
    }

    template <typename T>
    const T *operator()(const std::vector<T> &vec, const unsigned x, const unsigned y) const {
        return &vec[operator()(x, y)];
    }

    template <typename T>
    T *operator()(std::vector<T> &vec, const unsigned x, const unsigned y) const {
        return &vec[operator()(x, y)]; // too lazy to do the const cast trick
    }

};

/** converts relative coordinates to an edge index, edges are stored in { down, right, down, right ...} order */
struct edge_indexer {
    const uint width_, height_;
    explicit edge_indexer(const uint width, const uint height) : width_(width), height_(height) { }

    unsigned operator()(const uint x, const uint y, const move m) const {
        switch (m) {
            case move::UP:    return (x + width_ * (y - 1)) * 2;
            case move::DOWN:  return (x + width_ * y) * 2;
            case move::LEFT:  return ((x - 1) + width_ * y) * 2 + 1;
            case move::RIGHT: return (x + width_ * y) * 2 + 1;
        }
    }

    unsigned operator()(const uint idx, const move m) const {
        switch (m) {
            case move::UP:    return idx * 2 - width_ * 2;
            case move::DOWN:  return idx * 2;
            case move::LEFT:  return (idx - 1) * 2 + 1;
            case move::RIGHT: return idx * 2 + 1;
        }
    }
};

#endif // UTIL_H
