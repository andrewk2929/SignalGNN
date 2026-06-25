"""
network_generator.py

Generates a synthetic but biologically-motivated immune receptor signaling
network. Real curated signaling databases (Reactome, KEGG, InnateDB) require
licensed access or heavyweight parsers, so this module instead builds a graph
that mirrors the real *structural* properties of signaling networks:

  - Scale-free-ish degree distribution (a few highly-connected "hub" molecules,
    e.g. NF-kB, STAT3, mirroring real signaling hub topology)
  - Layered structure: receptors -> adaptor/kinase intermediates -> transcription
    factors -> output cytokines/effectors (mirrors real signal transduction
    cascades, e.g. TLR -> MyD88 -> IRAK -> NF-kB -> cytokine output)
  - Sparse connectivity (signaling networks are not densely connected)
  - Directed edges with signed weights (+1 activating, -1 inhibitory), since
    real immune signaling includes both activating and inhibitory crosstalk

This lets us simulate the core problem ImmuNet is solving: given a combination
of simultaneously activated upstream receptors, predict the downstream
molecular/cytokine output -- without needing a licensed real-world dataset.
"""

import numpy as np
import networkx as nx


# Named layers loosely inspired by real innate immune signaling biology.
# This is illustrative naming for realism/readability, not a literal
# reproduction of any specific curated pathway database.
LAYER_NAMES = {
    "receptor": [
        "TLR4", "TLR2", "TLR9", "TLR7", "TNFR1", "IL1R", "IFNAR1",
        "IFNAR2", "NLRP3", "RIG-I", "MDA5", "CD14",
    ],
    "adaptor": [
        "MyD88", "TRIF", "TRAF6", "TRAF3", "IRAK1", "IRAK4",
        "MAVS", "ASC", "RIP1", "FADD",
    ],
    "kinase": [
        "TAK1", "IKKa", "IKKb", "TBK1", "IKKe", "JAK1", "JAK2",
        "TYK2", "MAPK14", "MAP2K3",
    ],
    "tf": [
        "NF-kB", "IRF3", "IRF7", "STAT1", "STAT3", "AP-1", "NLRP3-inflammasome",
    ],
    "output": [
        "TNF-alpha", "IL-6", "IL-1beta", "IFN-beta", "IFN-alpha",
        "IL-10", "IL-12", "CXCL10", "ISG15",
    ],
}

LAYER_ORDER = ["receptor", "adaptor", "kinase", "tf", "output"]


def build_signaling_network(seed: int = 42, extra_skip_edges: int = 8,
                             inhibitory_fraction: float = 0.15) -> nx.DiGraph:
    """
    Build a directed, layered, signed signaling network.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    extra_skip_edges : int
        Number of additional "skip-layer" edges to add (e.g. receptor directly
        to TF), which is common in real crosstalk-heavy signaling networks.
    inhibitory_fraction : float
        Fraction of edges that are inhibitory (-1) rather than activating (+1).

    Returns
    -------
    G : networkx.DiGraph
        Directed graph with node attribute 'layer' and edge attribute 'weight'
        (+1 or -1).
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()

    # Add nodes with layer metadata
    for layer in LAYER_ORDER:
        for name in LAYER_NAMES[layer]:
            G.add_node(name, layer=layer)

    # Connect each layer to the next with a sparse, scale-free-ish pattern:
    # a few "hub" nodes in each downstream layer receive disproportionately
    # more incoming edges (mirrors real hub molecules like NF-kB).
    for i in range(len(LAYER_ORDER) - 1):
        src_layer = LAYER_NAMES[LAYER_ORDER[i]]
        dst_layer = LAYER_NAMES[LAYER_ORDER[i + 1]]

        # hub preference weights for destination nodes (some nodes are more
        # "popular" convergence points than others)
        hub_weights = rng.dirichlet(np.ones(len(dst_layer)) * 0.5)

        for src in src_layer:
            # each upstream node fans out to 1-3 downstream nodes
            n_targets = rng.integers(1, min(4, len(dst_layer) + 1))
            targets = rng.choice(
                dst_layer, size=n_targets, replace=False, p=hub_weights
            )
            for dst in targets:
                sign = -1 if rng.random() < inhibitory_fraction else 1
                G.add_edge(src, dst, weight=sign)

        # Guarantee every destination node has at least one incoming edge.
        # The random fan-out above can (and did, prior to this fix) leave
        # some downstream nodes with zero predecessors -- e.g. a cytokine
        # output that no upstream TF happens to target -- which makes that
        # node permanently stuck at zero activation regardless of input.
        # That's a generation bug, not a biological feature, so we patch it
        # here by connecting any orphaned destination node to 1-2 random
        # upstream nodes from the source layer.
        for dst in dst_layer:
            if G.in_degree(dst) == 0:
                n_fix = rng.integers(1, 3)
                fix_srcs = rng.choice(src_layer, size=min(n_fix, len(src_layer)),
                                       replace=False)
                for src in fix_srcs:
                    sign = -1 if rng.random() < inhibitory_fraction else 1
                    G.add_edge(src, dst, weight=sign)

    # Add some skip-layer edges for crosstalk realism (e.g. a kinase directly
    # affecting an output cytokine, bypassing the TF layer abstraction)
    all_nodes = list(G.nodes())
    layers = nx.get_node_attributes(G, "layer")
    layer_idx = {name: LAYER_ORDER.index(layers[name]) for name in all_nodes}

    added = 0
    attempts = 0
    while added < extra_skip_edges and attempts < extra_skip_edges * 20:
        attempts += 1
        src, dst = rng.choice(all_nodes, size=2, replace=False)
        if layer_idx[dst] > layer_idx[src] and not G.has_edge(src, dst):
            sign = -1 if rng.random() < inhibitory_fraction else 1
            G.add_edge(src, dst, weight=sign)
            added += 1

    return G


def get_layer_nodes(G: nx.DiGraph, layer: str):
    """Return list of node names belonging to a given layer."""
    return [n for n, d in G.nodes(data=True) if d["layer"] == layer]


if __name__ == "__main__":
    G = build_signaling_network()
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    for layer in LAYER_ORDER:
        print(f"  {layer}: {len(get_layer_nodes(G, layer))} nodes")
