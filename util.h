#ifndef UTIL_H
#define UTIL_H

#include <vector>

/** Makes indexing two dimensional arrays a bit easier */
struct indexer {
    const unsigned width_, height_, scale_;
    explicit indexer(const unsigned width, const unsigned height, const unsigned scale = 1) : width_(width), height_(height), scale_(scale) { }

    /** returns the index in a 1d array */
    unsigned operator()(const unsigned x, const unsigned y) const {
        return scale_ * (x + width_ * y);
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
#endif // UTIL_H
