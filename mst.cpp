#include <algorithm>
#include <tuple>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <queue>

#include <cassert>

#include "util.h"

namespace {
    typedef unsigned int uint;
    typedef unsigned char uchar;

    const uint special_idx = std::numeric_limits<uint>::max();

    // adjacency list to represent the grid, since it isn't densely connected
    union vertex {
        struct {
            uint neighbours[4]; // four neighbours
        };
   };
}

std::vector<float> sample_edge_apparence(const uint width, const uint height, const uint max_iter) {
    // create minimal spanning trees with the edges having random weights [0,1], until all the edges are covered, count edge apparences
    const indexer ndx(width, height);
    const edge_indexer edx(width, height);

    // make the graph
    const uint node_count = width * height;

    std::unordered_set<uint> specials;
    std::unordered_map<uint, std::tuple<uint, uint>> special_edge;

    // top and bottom row
    for (uint x = 1; x < width - 1; ++x) {
        const uint top = ndx(x, 0), bottom = ndx(x, height - 1);
        specials.insert(top);
        specials.insert(bottom);

        special_edge[top] = std::make_tuple(edx(top, move::DOWN), ndx(top, move::DOWN));
        special_edge[bottom] = std::make_tuple(edx(bottom, move::UP), ndx(bottom, move::UP));
    }

    // left and right columns
    for (uint y = 1; y < height - 1; ++y) {
        const uint left = ndx(0, y), right = ndx(width - 1, y);
        specials.insert(left);
        specials.insert(right);

        special_edge[left] = std::make_tuple(edx(left, move::RIGHT), ndx(left, move::RIGHT));
        special_edge[right] = std::make_tuple(edx(right, move::LEFT), ndx(right, move::LEFT));
    }

    // record which edges were chosen
    std::vector<float> edge_record(node_count * 2);

    for (uint i = 0; i < max_iter; ++i) {
        std::priority_queue<std::tuple<int, uint, uint>> heap; // <weight, edge, other_node>
        std::vector<uchar> marked(node_count);
        uint marked_count = 4; // special case the corners

        auto add_node = [&heap, &marked, &marked_count](const uint edge, const uint to) {
            // the edge weight is computed here
            heap.push(std::make_tuple(rand(), edge, to));
        };

        auto add_adjacent = [&specials, &special_edge, &heap, &edx, &ndx, &add_node](const uint node) {
            if (specials.find(node) == specials.end()) {
                for (uint j = 0; j < 4; ++j) {
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

        const uint inital = ndx(width / 2, height / 2);
        add_adjacent(inital);

        while (marked_count != node_count) {
            std::tuple<int, uint, uint> min_edge = heap.top();
            heap.pop();

            const uint node = std::get<2>(min_edge);
            if (!marked[node]) {
                marked[node] = 1;
                ++marked_count;

                add_adjacent(node);
                ++edge_record[std::get<1>(min_edge)];
            }
        }
    }

    std::transform(edge_record.begin(), edge_record.end(), edge_record.begin(), [max_iter](const float x) { return x / max_iter; });
    return edge_record;
}
