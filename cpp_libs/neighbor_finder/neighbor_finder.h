#ifndef NEIGHBOR_FINDER_LIBRARY_H
#define NEIGHBOR_FINDER_LIBRARY_H

#include <deque>
#include <unordered_map>
#include <tuple>
#include <cstdint>
#include <queue>

#include "thread_pool.h"

// Declaration of the NeighborInfo struct
struct NeighborInfo {
    int node;
    int neighbor;
    int64_t timestamp;
    int64_t edge_idx;
    std::shared_ptr<std::vector<float>> edge_features;

    NeighborInfo(
        int node,
        int neighbor,
        int64_t timestamp,
        int64_t edge_idx,
        std::shared_ptr<std::vector<float>> edge_features);

    bool operator<(const NeighborInfo& other) const;
};


class NeighborFinder {
private:
    int n_edge_features;
    bool uniform;
    int64_t neighbor_buff_duration_ms;
    std::unordered_map<int, std::deque<NeighborInfo>> adj_list;
    std::unordered_map<int, std::deque<NeighborInfo>> adj_list_snapshot;
    int64_t latest_timestamp;
    ThreadPool thread_pool;

public:
    // Constructor
    NeighborFinder(int neighbor_buffer_duration_hours, int n_edge_features, bool uniform = false);

    // Method to add interactions to the adjacency list
    void add_interactions(
        const std::vector<int>& upstreams,
        const std::vector<int>& downstreams,
        const std::vector<int64_t>& timestamps,
        const std::vector<int64_t>& edge_idxs,
        const std::vector<std::vector<float>>& edge_features);

    // Method to get temporal neighbors
    std::tuple<std::vector<int>, std::vector<int64_t>, std::vector<int64_t>, std::vector<float>, int>
    get_temporal_neighbor(const std::vector<int>& source_nodes, int n_neighbors);

    // Method to reset the adjacency list
    void reset();

    // Method to create a snapshot of the adjacency list
    void snapshot();

    // Method to restore the adjacency list from the snapshot
    void restore();
};


#endif //NEIGHBOR_FINDER_LIBRARY_H
