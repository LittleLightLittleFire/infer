#ifndef INFER_CRF_H
#define INFER_CRF_H

#include "util.h"

#include <vector>

namespace infer {

/**
 * Immutable class representing a grid conditional markov random field
 */
class crf {

public:
    /**
     * The type of the pairwise term of this CRF
     */
    enum class type {
        ARRAY, L1, L2
    };

    const unsigned width_, height_, labels_;
    const type type_;

    const std::vector<float> unary_;

    const float lambda_;
    const float trunc_;

private:
    const std::vector<float> pairwise_;

    const indexer idx_;
    const indexer ndx_;

public:
    /**
     * Initalise the grid CRF with a linear or quadratic potential
     *
     * @param width width of the CRF grid
     * @param height height of the CRF grid
     * @param labels number of labels in the CRF
     * @param potentials a width x height x label array of unary potentials
     * @param lambda scaling of the pairwise potentials
     * @param norm 1 or 2
     * @param trunc constant used to truncate the norm
     */
   explicit crf(const unsigned width, const unsigned height, const unsigned labels, const std::vector<float> unary, const float lambda, const unsigned norm, const unsigned truc);

    /**
     * Initalise the grid CRF with an array of labels x labels pairwise terms
     *
     * @param width width of the CRF grid
     * @param height height of the CRF grid
     * @param labels number of labels in the CRF
     * @param potentials a width x height x label array of unary potentials
     * @param lambda scaling of the pairwise potentials
     * @param potentials a label x label array of potentials
     */
    explicit crf(const unsigned width, const unsigned height, const unsigned labels, const std::vector<float> unary, const float lambda, const std::vector<float> pairwise);

    /**
     * Gets the unary potential given the node's coordinates and label
     */
    float unary(const unsigned x, const unsigned y, const unsigned label) const;

    /**
     * Gets the whole vector of unary potential for that node
     */
    const float *unary(const unsigned x, const unsigned y) const;

    /**
     * Gets the pairwise potential for two nodes given their labels, lambda is applied to the result
     */
    float pairwise(const unsigned x1, const unsigned y1, const float l1, const unsigned x2, const unsigned y2, const float l2) const;
};

}

#endif // INFER_CRF_H
