#include <algorithm>
#include <tuple>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <queue>

#include "util.h"

namespace infer {

std::vector<float> sample_edge_apparence(const unsigned width, const unsigned height, const unsigned max_iter) {
    // create minimal spanning trees with the edges having random weights [0,1], until all the edges are covered, count edge apparences
    const indexer ndx(width, height);
    const edge_indexer edx(width, height);

    // make the graph
    const unsigned node_count = width * height;

    std::unordered_set<unsigned> specials;
    std::unordered_map<unsigned, std::tuple<unsigned, unsigned>> special_edge;

    // top and bottom row
    for (unsigned x = 1; x < width - 1; ++x) {
        const unsigned top = ndx(x, 0), bottom = ndx(x, height - 1);
        specials.insert(top);
        specials.insert(bottom);

        special_edge[top] = std::make_tuple(edx(top, move::DOWN), ndx(top, move::DOWN));
        special_edge[bottom] = std::make_tuple(edx(bottom, move::UP), ndx(bottom, move::UP));
    }

    // left and right columns
    for (unsigned y = 1; y < height - 1; ++y) {
        const unsigned left = ndx(0, y), right = ndx(width - 1, y);
        specials.insert(left);
        specials.insert(right);

        special_edge[left] = std::make_tuple(edx(left, move::RIGHT), ndx(left, move::RIGHT));
        special_edge[right] = std::make_tuple(edx(right, move::LEFT), ndx(right, move::LEFT));
    }

    // record which edges were chosen
    std::vector<float> edge_record(node_count * 2);

    for (unsigned i = 0; i < max_iter; ++i) {
        std::priority_queue<std::tuple<int, unsigned, unsigned>> heap; // <weight, edge, other_node>
        std::vector<unsigned char> marked(node_count);
        unsigned marked_count = 4; // special case the corners

        auto add_node = [&heap, &marked, &marked_count](const unsigned edge, const unsigned to) {
            // the edge weight is computed here
            heap.push(std::make_tuple(rand(), edge, to));
        };

        auto add_adjacent = [&specials, &special_edge, &heap, &edx, &ndx, &add_node](const unsigned node) {
            if (specials.find(node) == specials.end()) {
                for (unsigned j = 0; j < 4; ++j) {
                    const unsigned edge = edx(node, static_cast<move>(j));
                    const unsigned other_node = ndx(node, static_cast<move>(j));

                    add_node(edge, other_node);
                }
            } else {
                const unsigned edge = std::get<0>(special_edge[node]);
                const unsigned other_node = std::get<1>(special_edge[node]);

                add_node(edge, other_node);
            }
        };

        const unsigned inital = ndx(width / 2, height / 2);
        add_adjacent(inital);

        while (marked_count != node_count) {
            std::tuple<int, unsigned, unsigned> min_edge = heap.top();
            heap.pop();

            const unsigned node = std::get<2>(min_edge);
            if (!marked[node]) {
                marked[node] = 1;
                ++marked_count;

                add_adjacent(node);
                ++edge_record[std::get<1>(min_edge)];
            }
        }
    }

    // find the average
    std::transform(edge_record.begin(), edge_record.end(), edge_record.begin(), [max_iter](const float x) { return x / max_iter; });

    return edge_record;
}

}
