#include "neighbor_finder.h"

#include <iostream>
#include <random>
#include <unordered_set>
#include <thread>

NeighborInfo::NeighborInfo(
    const int node,
    const int neighbor,
    const int64_t timestamp,
    const int64_t edge_idx,
    std::shared_ptr<std::vector<float>> edge_features)
        : node(node), neighbor(neighbor), timestamp(timestamp), edge_idx(edge_idx), edge_features(std::move(edge_features)) {}

bool NeighborInfo::operator<(const NeighborInfo &other) const {
    return this->timestamp < other.timestamp;
}

NeighborFinder::NeighborFinder(
    const int neighbor_buffer_duration_hours,
    const int n_edge_features,
    const bool uniform):
        n_edge_features(n_edge_features), uniform(uniform),
        neighbor_buff_duration_ms(neighbor_buffer_duration_hours * 60 * 60 * 1000),
        latest_timestamp(0),
        thread_pool(ThreadPool(std::thread::hardware_concurrency())) {}

void NeighborFinder::add_interactions(
    const std::vector<int>& upstreams,
    const std::vector<int>& downstreams,
    const std::vector<int64_t>& timestamps,
    const std::vector<int64_t>& edge_idxs,
    const std::vector<std::vector<float>>& edge_features) {

    std::unordered_map<int, std::vector<NeighborInfo>> batch_nodes;
    std::unordered_set<int> unique_node_set;

    for (const auto& node : upstreams) {
        unique_node_set.insert(node);
    }
    for (const auto& node : downstreams) {
        unique_node_set.insert(node);
    }

    const std::vector<int> unique_node_list(unique_node_set.begin(), unique_node_set.end());

    for (const auto& node : unique_node_list) {
        batch_nodes[node] = {};
    }

    if (!timestamps.empty()) {
        latest_timestamp = timestamps.back();
    }

    for (size_t i = 0; i < upstreams.size(); ++i) {
        auto shared_edge_features = std::make_shared<std::vector<float>>(edge_features[i]);

        // Create NeighborInfo objects
        NeighborInfo us_node(
            upstreams[i],
            downstreams[i],
            timestamps[i],
            edge_idxs[i],
            shared_edge_features
        );
        NeighborInfo ds_node(
            downstreams[i],
            upstreams[i],
            timestamps[i],
            edge_idxs[i],
            shared_edge_features
        );

        // Add nodes to batch_nodes
        batch_nodes[upstreams[i]].push_back(us_node);
        batch_nodes[downstreams[i]].push_back(ds_node);
    }

    // Update adj_list with batch_nodes
    for (const auto& [node_name, node_obj_list] : batch_nodes) {
        if (adj_list.find(node_name) == adj_list.end()) {
            adj_list[node_name] = std::deque<NeighborInfo>();
        }

        auto& node_adj_list = adj_list[node_name];
        node_adj_list.insert(node_adj_list.end(), node_obj_list.begin(), node_obj_list.end());

        // Remove outdated entries based on timestamp
        while (!node_adj_list.empty() &&
               latest_timestamp - node_adj_list.front().timestamp > neighbor_buff_duration_ms) {
            node_adj_list.pop_front();
        }
    }
}

std::tuple<std::vector<int>, std::vector<int64_t>, std::vector<int64_t>, std::vector<float>, int>
NeighborFinder::get_temporal_neighbor(const std::vector<int>& source_nodes, const int n_neighbors) {
    const size_t num_source_nodes = source_nodes.size();

    std::vector<int> neighbors(num_source_nodes * n_neighbors);
    std::vector<int64_t> edge_indices(num_source_nodes * n_neighbors);
    std::vector<int64_t> timestamps(num_source_nodes * n_neighbors);
    std::vector<float> edge_features(num_source_nodes * n_neighbors * n_edge_features);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::thread> threads;

    // Lambda function for thread work
    auto process_node = [&](const size_t i) {
        auto write_idx = static_cast<int>(i) * n_neighbors;

        const auto source_node = source_nodes[i];
        const auto it = adj_list.find(source_node);
        if (it == adj_list.end()) return 0;

        std::deque<NeighborInfo>& source_adj = it->second;
        const int derived_n_neighbors = std::min(n_neighbors, static_cast<int>(source_adj.size()));
        std::vector<NeighborInfo> entries;

        if (uniform && !source_adj.empty()) {
            std::uniform_int_distribution<> dis(0, static_cast<int>(source_adj.size()) - 1);
            for (int j = 0; j < n_neighbors; ++j) {
                entries.push_back(source_adj[dis(gen)]);
            }
        } else {
            entries.insert(entries.end(), source_adj.end() - derived_n_neighbors, source_adj.end());
        }

        for (const NeighborInfo& entry : entries) {
            neighbors[write_idx] = entry.neighbor;
            edge_indices[write_idx] = entry.edge_idx;
            timestamps[write_idx] = entry.timestamp;

            std::copy(
                entry.edge_features->begin(),
                entry.edge_features->end(),
                edge_features.begin() + write_idx * n_edge_features);

            ++write_idx;
        }

        return 1;
    };

    std::vector<std::future<int>> results;

    // Launch threads
    for (size_t i = 0; i < num_source_nodes; ++i) {
        results.emplace_back(thread_pool.enqueue(process_node, i));
    }

    // Wait for all futures to finish
    for (auto& future : results) {
        future.wait();
    }

    return std::make_tuple(
        neighbors,
        edge_indices,
        timestamps,
        edge_features,
        n_edge_features);
}

void NeighborFinder::reset() {
    adj_list.clear();
}

void NeighborFinder::snapshot() {
    adj_list_snapshot = adj_list;
}

void NeighborFinder::restore() {
    if (!adj_list_snapshot.empty()) {
        adj_list = adj_list_snapshot;
    }
}


