#include "driver.h"

#include "bp.h"
#include "trbp.h"
#include "mst.h"
#include "qp.h"
#include "compose.h"

#ifdef GPU_SUPPORT
#include "cuda/bp.h"
#include "cuda/trbp.h"
#endif

#include <string>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <memory>

namespace {
    const unsigned mst_samples = 200;

    const unsigned hbp_layers = 5;
    const unsigned rounds_per_layer = 10;

    const std::string help_string =
        "./driver [algorithm] (-v) (-r rounds) [example] ...\n"
        "    algorithm\n"
        "        bp, bp_async, trbp, trbp_async, hbp, hbp_async, gpu_bp, gpu_trbp, or gpu_hbp\n"
        "    example\n"
        "        stereo, iseg, or restore\n"
        "    -v verbose (optional)\n"
        "        enables verbose output, prints energy per round\n"
        "    -r rounds (optional)\n"
        "        number of rounds to run, default 50 rounds, 5x10 rounds for hbp\n\n"
        " additional options are required depeneding on the example: \n"
        "    ./driver [algorithm] stereo [labels] [scale] [left_image.png] [right_image.png] [out_image.png]\n"
        "    ./driver [algorithm] iseg [image.png] [annotation.png] [out_image.png]\n"
        "    ./driver [algorithm] restore [image.png] [out_image.png]\n";

    void output_energy(const std::string name, const std::vector<unsigned> result, const infer::crf &crf, const std::string round) {
        const float unary = crf.unary_energy(result);
        const float pairwise = crf.pairwise_energy(result);
        std::cout << name << " " << round << " " << unary + pairwise << " " << unary << " " << pairwise << std::endl;
    }

    template <class M>
    void runner(const std::shared_ptr<M> m, const infer::crf &crf, const unsigned rounds, const bool verbose) {
        if (verbose) {
            for (unsigned i = 0; i < rounds; ++i) {
                m->run(1);
                output_energy(m->get_name(), m->get_result(), crf, std::to_string(i));
            }
        } else {
            m->run(rounds);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << help_string << std::endl;
        return 1;
    }

    // load arguments
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
    }

    const std::string algorithm = args[1];
    unsigned offset = 2; // starting of the example arguments

    unsigned rounds = 50;
    bool verbose = false;

    { // set verbose
        const auto vswitch = std::find(args.cbegin(), args.cend(), "-v");
        if (vswitch != args.cend()) {
            verbose = true;
            ++offset;
        }
    }

    { // set rounds
        const auto rswitch = std::find(args.cbegin(), args.cend(), "-r");
        if (rswitch < args.cend() - 1) {
            rounds = std::stoi(*(rswitch + 1));
            offset += 2;
        }
    }

    const std::string example = args[offset];

    // give a CRF, return the result using the chosen algorithm
    std::function<const std::vector<unsigned>(const infer::crf)> method = [&algorithm, rounds, verbose](const infer::crf crf) {
        if (algorithm.substr(0, 3) != "gpu") {
            // methods based on composition
            if (algorithm == "hbp") {
                const std::vector<unsigned> result = infer::compose<infer::bp>(hbp_layers, rounds_per_layer, crf, [](const infer::crf &downsized) { return infer::bp(downsized, true); });
                if (verbose) {
                    output_energy("hbp", result, crf, std::to_string(hbp_layers) + "x" + std::to_string(rounds_per_layer));
                }
            } else if (algorithm == "hbp_async") {
                const std::vector<unsigned> result = infer::compose<infer::bp>(hbp_layers, rounds_per_layer, crf, [](const infer::crf &downsized) { return infer::bp(downsized, false); });
                if (verbose) {
                    output_energy("hbp_async", result, crf, std::to_string(hbp_layers) + "x" + std::to_string(rounds_per_layer));
                }
            }

            // normal methods
            std::shared_ptr<infer::method> method;

            if (algorithm == "bp") {
                method = std::shared_ptr<infer::method>(new infer::bp(crf, true));
            } else if (algorithm == "bp_async") {
                method = std::shared_ptr<infer::method>(new infer::bp(crf, false));
            } else if (algorithm == "trbp") {
                method = std::shared_ptr<infer::method>(new infer::trbp(crf, infer::sample_edge_apparence(crf.width_, crf.height_, mst_samples), true));
            } else if (algorithm == "trbp_async") {
                method = std::shared_ptr<infer::method>(new infer::trbp(crf, infer::sample_edge_apparence(crf.width_, crf.height_, mst_samples), false));
            } else if (algorithm == "qp") {
                method = std::shared_ptr<infer::method>(new infer::qp(crf));
            }

            if (method) {
                runner(method, crf, rounds, verbose);
                return method->get_result();
            } else {
                throw std::runtime_error("Unknown algorithm: " + algorithm);
            }
        } else {
#ifdef GPU_SUPPORT
            infer::cuda::crf gpu_crf = crf.to_gpu();
            std::shared_ptr<infer::cuda::method> gpu_method;

            if (algorithm == "gpu_bp") {
                gpu_method = std::shared_ptr<infer::cuda::method>(new infer::cuda::bp(gpu_crf));
            } else if (algorithm == "gpu_trbp") {
                gpu_method = std::shared_ptr<infer::cuda::method>(new infer::cuda::trbp(gpu_crf, infer::sample_edge_apparence(crf.width_, crf.height_, mst_samples)));
            }

            if (gpu_method) {
                runner(gpu_method, crf, rounds, verbose);
                return gpu_method->get_result();
            } else {
                throw std::runtime_error("Unknown GPU algorithm: " + algorithm);
            }
#else
            throw std::runtime_error("GPU_SUPPORT was not switched on during build");
#endif
        }

    };

    if (example == "stereo") {
        stereo::run(method, std::stoi(args[offset+1]), std::stoi(args[offset+2]), args[offset+3], args[offset+4], args[offset+5]);
    } else if (example == "iseg") {
        if (argc < 6) {
            std::cerr << "Not enough arguments for iseg" << std::endl;
            return 1;
        }
        iseg::run(method, args[offset+1], args[offset+2], args[offset+3]);
    } else if (example == "restore") {
    } else {
        std::cerr << "Unknown example: " << example << std::endl;
    }

    return 0;
}
