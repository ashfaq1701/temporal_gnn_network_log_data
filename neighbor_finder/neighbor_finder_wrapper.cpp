#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "neighbor_finder.h"

namespace py = pybind11;

PYBIND11_MODULE(neighbor_finder, m) {
    py::class_<NeighborFinder>(m, "NeighborFinder")
        .def(py::init<int, int, bool>())
        .def("add_interactions", &NeighborFinder::add_interactions)
        .def("get_temporal_neighbor", [](NeighborFinder& nf, const std::vector<int>& source_nodes, int n_neighbors) {
            auto [neighbors, edge_indices, timestamps, edge_features, n_edge_feats] =
                nf.get_temporal_neighbor(source_nodes, n_neighbors);

            int num_source_nodes = static_cast<int>(source_nodes.size());

            py::array_t<int> py_neighbors(
                {num_source_nodes, n_neighbors},
                neighbors.data());

            // Create py::array_t for edge_indices
            py::array_t<int64_t> py_edge_indices(
                {num_source_nodes, n_neighbors},
                edge_indices.data());

            // Create py::array_t for timestamps
            py::array_t<int64_t> py_timestamps(
                {num_source_nodes, n_neighbors},
                timestamps.data());

            py::array_t<float> py_edge_features(
                {num_source_nodes, n_neighbors, n_edge_feats},
                edge_features.data());

            return std::make_tuple(py_neighbors, py_edge_indices, py_timestamps, py_edge_features);
        })
        .def("reset", &NeighborFinder::reset)
        .def("snapshot", &NeighborFinder::snapshot)
        .def("restore", &NeighborFinder::restore);
}
