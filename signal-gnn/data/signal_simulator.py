"""
signal_simulator.py

Simulates signal propagation through the immune signaling network to generate
(input activation pattern -> output activation pattern) training pairs.

This is the core data-generating process that mirrors what ImmuNet's GNN is
ultimately trying to learn: given which receptors are simultaneously
activated, what is the resulting molecular/cytokine output?

Propagation model
------------------
We use a simple discrete-time signal propagation model on the DAG-like
layered network:

  1. A random subset of receptors is set to "active" (this is the input --
     analogous to "simultaneous immune receptor activations").
  2. Activation propagates layer by layer. A downstream node's activation is
     a (saturating) function of the weighted sum of its active upstream
     neighbors' activations, with signed edge weights (+1 activating,
     -1 inhibitory) and a small amount of stochastic noise (mirroring
     biological variability).
  3. The final "output" layer activations are the label the GNN must learn
     to predict from the receptor-layer input.

This gives us a fully self-contained, reproducible dataset without requiring
a licensed real-world immunology dataset, while preserving the key structural
problem: predicting combinatorial downstream effects of simultaneous upstream
activations.
"""

import numpy as np
import networkx as nx

from network_generator import LAYER_ORDER, get_layer_nodes


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_one(G: nx.DiGraph, node_index: dict, active_receptors: set,
                  rng: np.random.Generator, noise_std: float = 0.08,
                  gain: float = 2.4, bias: float = -0.9) -> np.ndarray:
    """
    Propagate a single receptor-activation pattern through the network.

    Parameters
    ----------
    G : networkx.DiGraph
        The signaling network.
    node_index : dict
        Mapping from node name -> integer index.
    active_receptors : set
        Set of receptor node names that are "on" for this sample.
    rng : np.random.Generator
    noise_std : float
        Std of Gaussian noise added to each node's activation (biological
        variability).
    gain : float
        Scales the normalized weighted input before the sigmoid squashing
        function; higher gain -> sharper activation thresholds.
    bias : float
        Constant offset applied before the sigmoid. Negative bias means a
        node needs a real fraction of its predecessors active (not just one
        out of many) to turn on -- this keeps high-in-degree nodes (e.g. a
        cytokine with 6 upstream TFs) from saturating to ~1.0 regardless of
        *which* specific receptors fired, which is what happened before this
        was added.

    Returns
    -------
    activation : np.ndarray, shape (num_nodes,)
        Activation level (0-1) of every node in the network, in node_index
        order.
    """
    n = G.number_of_nodes()
    activation = np.zeros(n, dtype=np.float32)

    for r in active_receptors:
        activation[node_index[r]] = 1.0

    # Propagate layer by layer (receptor -> adaptor -> kinase -> tf -> output)
    for layer in LAYER_ORDER[1:]:
        for node in get_layer_nodes(G, layer):
            idx = node_index[node]
            preds = list(G.predecessors(node))
            if not preds:
                continue
            weighted_sum = 0.0
            for p in preds:
                w = G[p][node]["weight"]
                weighted_sum += w * activation[node_index[p]]
            # Normalize by in-degree so nodes with many predecessors don't
            # trivially saturate just from having more inputs to sum over --
            # what matters is what *fraction* of a node's inputs are active,
            # not the raw count.
            normalized_input = weighted_sum / len(preds)
            noise = rng.normal(0, noise_std)
            activation[idx] = float(
                sigmoid(gain * normalized_input + bias + noise)
            )

    return activation


def generate_dataset(G: nx.DiGraph, num_samples: int = 2000, seed: int = 0,
                      min_active: int = 1, max_active: int = 4):
    """
    Generate a full dataset of (receptor activation pattern -> full-network
    activation) pairs.

    Returns
    -------
    node_index : dict
        node name -> integer index (consistent ordering across all samples)
    X_receptor : np.ndarray, shape (num_samples, num_receptor_nodes)
        Binary receptor activation patterns (the "input": which receptors
        fired simultaneously).
    Y_full : np.ndarray, shape (num_samples, num_nodes)
        Full-network activation state resulting from propagating each input.
    receptor_names : list[str]
    all_node_names : list[str]
    """
    rng = np.random.default_rng(seed)
    all_node_names = list(G.nodes())
    node_index = {name: i for i, name in enumerate(all_node_names)}
    receptor_names = get_layer_nodes(G, "receptor")

    X_receptor = np.zeros((num_samples, len(receptor_names)), dtype=np.float32)
    Y_full = np.zeros((num_samples, len(all_node_names)), dtype=np.float32)

    for s in range(num_samples):
        n_active = rng.integers(min_active, max_active + 1)
        active = set(rng.choice(receptor_names, size=n_active, replace=False))

        for j, r in enumerate(receptor_names):
            X_receptor[s, j] = 1.0 if r in active else 0.0

        Y_full[s] = simulate_one(G, node_index, active, rng)

    return node_index, X_receptor, Y_full, receptor_names, all_node_names


if __name__ == "__main__":
    from network_generator import build_signaling_network

    G = build_signaling_network()
    node_index, X, Y, receptor_names, all_names = generate_dataset(
        G, num_samples=5
    )
    print("Receptor names:", receptor_names)
    print("X shape:", X.shape, "Y shape:", Y.shape)
    print("Sample 0 active receptors:",
          [r for j, r in enumerate(receptor_names) if X[0, j] == 1.0])
    output_idx = [all_names.index(n) for n in
                  get_layer_nodes(G, "output")]
    print("Sample 0 output activations:",
          {all_names[i]: round(float(Y[0, i]), 3) for i in output_idx})
