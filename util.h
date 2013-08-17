#ifndef UTIL_H
#define UTIL_H

#include <vector>

enum class move { UP, DOWN, LEFT, RIGHT };

/** Makes indexing two dimensional arrays a bit easier */
struct indexer {
    const unsigned width_, height_;
    explicit indexer(const unsigned width, const unsigned height) : width_(width), height_(height) { }

    /** returns the index in a 1d array */
    unsigned operator()(const unsigned x, const unsigned y) const {
        return x + width_ * y;
    }

    template <typename T>
    const T &operator()(const std::vector<T> &vec, const unsigned x, const unsigned y) const {
        return vec[operator()(x, y)];
    }

    template <typename T>
    T &operator()(std::vector<T> &vec, const unsigned x, const unsigned y) const {
        return vec[operator()(x, y)]; // too lazy to do the const cast trick
    }

};

#endif // UTIL_H
