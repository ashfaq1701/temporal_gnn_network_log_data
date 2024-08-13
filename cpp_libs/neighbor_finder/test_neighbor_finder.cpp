#include <iostream>

#include "neighbor_finder.h"
using namespace std;

int main() {
    const auto neighbor_finder = new NeighborFinder(6, 5, false);
    neighbor_finder->add_interactions(
        {1, 2, 3, 4},
        {2, 4, 6, 7},
        {0, 2, 7, 8},
        {1, 2, 3, 4},
        {
            {0.0, 1.0, 0.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 0.0, 0.0},
            {0.0, 1.0, 1.0, 0.0, 0.0},
            {1.0, 1.0, 1.0, 0.0, 1.0}
        }
    );

    auto [neighbors, edge_indices, timestamps, edge_features, n_edge_feats] =
        neighbor_finder->get_temporal_neighbor(
        {2, 4, 9},
        5);

    std::cout << "Neighbors:\n";
    for (size_t i = 0; i < neighbors.size(); ++i) {
        std::cout << "Index " << i
                  << ": Neighbor = " << neighbors[i]
                  << ", Edge Index = " << edge_indices[i]
                  << ", Timestamp = " << timestamps[i]
                  << ", Edge Feature = ";

        // Print edge features
        for (int j = 0; j < 5; ++j) {
            std::cout << edge_features[i * 5 + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Num edge features: " << n_edge_feats << endl;
}
