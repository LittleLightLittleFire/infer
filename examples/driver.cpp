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

    const unsigned layers = 5;
    const unsigned rounds_per_layer = 10;

    const std::string help_string =
        "./driver [algorithm] (-a) (-v) (-r rounds) [example] ...\n"
        "    algorithm\n"
        "        bp, trbp, hbp, trhbp, gpu_bp, gpu_trbp, or gpu_hbp\n"
        "    example\n"
        "        stereo, iseg, or restore\n"
        "    -a async (optional)\n"
        "        run async version of the algorithm (slower, but better final result)\n"
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

                if (std::shared_ptr<infer::qp> qp = std::dynamic_pointer_cast<infer::qp>(m)) {
                    std::cout << qp->objective() << " ";
                }

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

    auto get_switch = [&offset, &args](const std::string s) {
        const bool result = std::find(args.cbegin(), args.cend(), s) != args.cend();
        if (result) {
            ++offset;
        }
        return result;
    };

    // set algorithm options
    const bool verbose = get_switch("-v");
    const bool sync = !get_switch("-a");
    unsigned rounds = 50;

    { // set number rounds
        const auto rswitch = std::find(args.cbegin(), args.cend(), "-r");
        if (rswitch < args.cend() - 1) {
            rounds = std::stoi(*(rswitch + 1));
            offset += 2;
        }
    }

    const std::string example = args[offset];

    // give a CRF, return the result using the chosen algorithm
    std::function<const std::vector<unsigned>(const infer::crf)> method = [&algorithm, rounds, verbose, sync](const infer::crf crf) {
        if (algorithm.substr(0, 3) != "gpu") {
            // methods based on composition
            if (algorithm == "hbp") {
                const std::vector<unsigned> result = infer::hbp(layers, rounds, sync, crf);
                if (verbose) {
                    output_energy(algorithm + (!sync ? "_async" : "") , result, crf, std::to_string(layers) + "x" + std::to_string(rounds_per_layer));
                }
                return result;
            } else if (algorithm == "trhbp") {
                const std::vector<unsigned> result = infer::trhbp(layers, rounds, infer::sample_edge_apparence(crf.width_, crf.height_, mst_samples, layers), sync, crf);
                if (verbose) {
                    output_energy(algorithm + (!sync ? "_async" : "") , result, crf, std::to_string(layers) + "x" + std::to_string(rounds_per_layer));
                }
                return result;
            }

            // normal methods
            std::shared_ptr<infer::method> method;

            if (algorithm == "bp") {
                method = std::shared_ptr<infer::method>(new infer::bp(crf, sync));
            } else if (algorithm == "trbp") {
                method = std::shared_ptr<infer::method>(new infer::trbp(crf, infer::sample_edge_apparence(crf.width_, crf.height_, mst_samples), sync));
            } else if (algorithm == "qp") {
                method = std::shared_ptr<infer::method>(new infer::qp(crf));
            }

            if (method) {
                runner(method, crf, rounds, verbose);
                return static_cast<const std::vector<unsigned>>(method->get_result());
            } else {
                throw std::runtime_error("Unknown algorithm: " + algorithm);
            }
        } else {
#ifdef GPU_SUPPORT

            // construct the corresponding CRF for the GPU
            infer::cuda::crf gpu_crf = [&crf](){
                if (crf.type_ == crf::type::ARRAY) {
                    return cuda::crf(crf.width_, crf.height_, crf.labels_, crf.unary_, crf.lambda_, crf.pairwise_);
                } else {
                    const unsigned norm = type_ == crf::type::L1 ? 1 : 2;
                    return cuda::crf(crf.width_, crf.height_, crf.labels_, crf.unary_, crf.lambda_, norm, crf.trunc_);
                }
            }();

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
