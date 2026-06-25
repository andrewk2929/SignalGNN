"""
pyg_dataset.py

Converts the simulated signaling data into PyTorch Geometric `Data` objects.

Framing as a graph-learning problem
------------------------------------
Each sample is the *same* underlying signaling network (same nodes, same
edges), but with a different set of receptor nodes turned "on". So rather
than one big graph per sample, we represent this as:

  - A single fixed graph topology (edge_index, edge weights) shared across
    all samples.
  - Per-sample *node features*: a one-hot "is this receptor active in this
    sample" flag, plus a static one-hot encoding of which layer the node
    belongs to (receptor / adaptor / kinase / tf / output). This lets the
    GNN learn "if receptor X is active, how does that propagate through the
    graph structure to affect node Y's output" -- which is exactly analogous
    to ImmuNet's "simultaneous receptor activation -> downstream molecular
    output" problem.
  - A per-sample *target*: the ground-truth activation level of every node
    (we'll mainly evaluate on the output-layer nodes, since that's the
    biologically meaningful prediction target).

This setup is sometimes called "inductive node regression with varying
input signals on a fixed graph," and it's a common pattern in systems
biology / signaling network ML.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from network_generator import LAYER_ORDER, build_signaling_network, get_layer_nodes
from signal_simulator import generate_dataset


def build_static_graph_tensors(G, all_node_names):
    """Build the fixed edge_index / edge_weight / layer-one-hot tensors that
    are shared across every sample (since the topology doesn't change)."""
    node_index = {name: i for i, name in enumerate(all_node_names)}

    src, dst, weights = [], [], []
    for u, v, data in G.edges(data=True):
        src.append(node_index[u])
        dst.append(node_index[v])
        weights.append(data["weight"])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    # Static layer one-hot feature per node (5 layers)
    layer_to_onehot = {layer: i for i, layer in enumerate(LAYER_ORDER)}
    layer_feat = torch.zeros((len(all_node_names), len(LAYER_ORDER)))
    layers = dict(G.nodes(data="layer"))
    for i, name in enumerate(all_node_names):
        layer_feat[i, layer_to_onehot[layers[name]]] = 1.0

    return edge_index, edge_weight, layer_feat


def make_pyg_samples(G, X_receptor, Y_full, receptor_names, all_node_names):
    """
    Build a list of torch_geometric Data objects, one per simulated sample.

    Node feature vector per node = [is_active (0/1), layer_onehot (5-dim)]
      -> 6-dim total node feature.
    Target (data.y) = activation value for every node (regression target),
      shape (num_nodes,).
    """
    edge_index, edge_weight, layer_feat = build_static_graph_tensors(
        G, all_node_names
    )
    node_index = {name: i for i, name in enumerate(all_node_names)}
    receptor_idx = [node_index[r] for r in receptor_names]

    samples = []
    num_samples = X_receptor.shape[0]
    num_nodes = len(all_node_names)

    for s in range(num_samples):
        is_active = torch.zeros((num_nodes, 1), dtype=torch.float32)
        for j, ridx in enumerate(receptor_idx):
            is_active[ridx, 0] = float(X_receptor[s, j])

        x = torch.cat([is_active, layer_feat], dim=1)  # (num_nodes, 6)
        y = torch.tensor(Y_full[s], dtype=torch.float32)  # (num_nodes,)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight.unsqueeze(1),
            y=y,
        )
        samples.append(data)

    return samples


def get_dataset(num_samples=2000, seed=0):
    """Convenience function: build network + simulate + convert to PyG."""
    G = build_signaling_network(seed=42)
    node_index, X, Y, receptor_names, all_node_names = generate_dataset(
        G, num_samples=num_samples, seed=seed
    )
    samples = make_pyg_samples(G, X, Y, receptor_names, all_node_names)
    output_node_idx = [all_node_names.index(n) for n in
                        get_layer_nodes(G, "output")]
    return {
        "graph": G,
        "samples": samples,
        "all_node_names": all_node_names,
        "receptor_names": receptor_names,
        "output_node_idx": output_node_idx,
        "num_node_features": samples[0].x.shape[1],
    }


if __name__ == "__main__":
    ds = get_dataset(num_samples=5)
    print("Num samples:", len(ds["samples"]))
    print("Node feature dim:", ds["num_node_features"])
    print("Example sample:", ds["samples"][0])
    print("Output node indices:", ds["output_node_idx"])
