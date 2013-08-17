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

    /** returns the node index after applying the operation */
    unsigned operator()(const unsigned idx, const move m) const {
        switch (m) {
            case move::UP:    return idx - width_;
            case move::DOWN:  return idx + width_;
            case move::LEFT:  return idx - 1;
            case move::RIGHT: return idx + 1;
        }
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
