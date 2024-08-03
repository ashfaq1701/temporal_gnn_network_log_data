from src.preprocess.functions import get_node_label_encoder, get_graphs


def filter_nodes_k_neighbors(nodes, k):
    label_encoder = get_node_label_encoder()
    downstream_graph, upstream_graph = get_graphs()
    encoded_nodes = label_encoder.transform(nodes)

    combined_set = set()

    for node in encoded_nodes:
        i_node_set, u_node_set = get_k_neighbor_sets(node, k, downstream_graph, upstream_graph)
        combined_set = combined_set | i_node_set | u_node_set

    node_list = list(combined_set)
    node_labels = label_encoder.inverse_transform(node_list)
    return node_labels


def get_k_neighbor_sets(node, k, downstream_graph, upstream_graph):
    i_node_set = set()
    u_node_set = set()

    def get_nested_k_neighbor_sets(graph, current_node, current_k, node_set):
        if current_k == 0:
            return

        node_set.add(current_node)

        neighbors = graph.get(current_node, {})
        for neighbor, _ in neighbors.items():
            get_nested_k_neighbor_sets(graph, neighbor, current_k - 1, node_set)

    get_nested_k_neighbor_sets(downstream_graph, node, k, i_node_set)
    get_nested_k_neighbor_sets(upstream_graph, node, k, u_node_set)
    return i_node_set, u_node_set
