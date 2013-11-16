#ifndef INFER_CUDA_CRF_H
#define INFER_CUDA_CRF_H

namespace infer {
namespace cuda {

/**
 * A conditional random field initalised on GPU memory
 */
class crf {
public:
    enum type {
        ARRAY, L1, L2
    };

    float *dev_unary_;
    const unsigned width_, height_;
    const type type_;

    const float lambda_;
    const float trunc_;
    float *dev_pairwise_;

    explicit crf(const unsigned width, const unsigned height, const std::vector<float> unary, const float lambda, const unsigned norm, const unsigned trunc);
    explicit crf(const unsigned width, const unsigned height, const std::vector<float> unary, const float lambda, const std::vector<float> pairwise);

private:
    // disable copy and move
    crf(const crf &);
    crf &operator=(const crf &);

    ~crf();
};

}
}

#endif // INFER_CUDA_CRF_H
