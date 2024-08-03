import os
import pandas as pd


def compute_downstream_graph_for_file(idx):
    input_dir = os.getenv('FINAL_DATA_DIR')
    filepath = os.path.join(input_dir, f'data_{idx}.parquet')
    df = pd.read_parquet(filepath)
    df_counts = df.groupby(['u', 'i']).size().reset_index(name='counts')

    downstream_graphs = {}
    upstream_graphs = {}

    for _, row in df_counts.iterrows():
        u = int(row['u'])
        i = int(row['i'])
        counts = int(row['counts'])

        downstream_counts = downstream_graphs.get(i, {})
        downstream_counts[u] = downstream_counts.get(u, 0) + counts
        downstream_graphs[i] = downstream_counts

        upstream_counts = upstream_graphs.get(u, {})
        upstream_counts[u] = upstream_counts.get(u, 0) + counts
        upstream_graphs[i] = upstream_counts

    print(f'Processed file {filepath}')
    return idx, downstream_graphs, upstream_graphs
