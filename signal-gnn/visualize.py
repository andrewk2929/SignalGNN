"""
visualize.py

Generates figures for the project:
  1. signaling_network.png   -- the layered network topology
  2. training_curves.png     -- train/val loss + output-node MAE over epochs
  3. prediction_scatter.png  -- predicted vs. true activation for output nodes
  4. attention_heatmap.png   -- learned attention weights for one example
                                 input, showing which upstream nodes mattered
                                 most for a given output prediction

Run after train.py has produced models/best_model.pt and
outputs/training_history.json.
"""

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch

from network_generator import build_signaling_network, LAYER_ORDER
from pyg_dataset import get_dataset
from gnn_model import SignalingGNN

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LAYER_COLORS = {
    "receptor": "#4C72B0",
    "adaptor": "#55A868",
    "kinase": "#C44E52",
    "tf": "#8172B2",
    "output": "#CCB974",
}


def plot_network(G):
    pos = {}
    layer_x = {layer: i for i, layer in enumerate(LAYER_ORDER)}
    layer_counts = {layer: 0 for layer in LAYER_ORDER}
    layer_totals = {layer: sum(1 for _, d in G.nodes(data=True)
                                if d["layer"] == layer) for layer in LAYER_ORDER}

    for node, data in G.nodes(data=True):
        layer = data["layer"]
        i = layer_counts[layer]
        total = layer_totals[layer]
        y = (i - (total - 1) / 2) * 1.2
        pos[node] = (layer_x[layer] * 4, y)
        layer_counts[layer] += 1

    fig, ax = plt.subplots(figsize=(16, 10))

    for u, v, data in G.edges(data=True):
        color = "#999999" if data["weight"] > 0 else "#E07A5F"
        style = "-" if data["weight"] > 0 else "--"
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(arrowstyle="-|>", color=color, alpha=0.45,
                             linestyle=style, lw=1.1,
                             shrinkA=12, shrinkB=12),
        )

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        color = LAYER_COLORS[data["layer"]]
        ax.scatter(x, y, s=900, c=color, edgecolors="black",
                   linewidths=1.2, zorder=3)
        ax.text(x, y, node, ha="center", va="center", fontsize=7,
                zorder=4, fontweight="bold", color="white")

    for layer, x in layer_x.items():
        ax.text(x * 4, max(pos[n][1] for n in G.nodes()
                            if G.nodes[n]["layer"] == layer) + 1.5,
                layer.upper(), ha="center", fontsize=13, fontweight="bold")

    ax.set_title(
        "Synthetic Immune Receptor Signaling Network\n"
        "(solid = activating edge, dashed red = inhibitory edge)",
        fontsize=14,
    )
    ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "signaling_network.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_training_curves():
    history_path = os.path.join(
        os.path.dirname(__file__), "outputs", "training_history.json"
    )
    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_mae = [h["val_output_mae"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="Train loss (MSE)")
    axes[0].plot(epochs, val_loss, label="Val loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_mae, color="#C44E52")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (output-layer nodes)")
    axes[1].set_title("Validation MAE on Output (Cytokine) Nodes")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_prediction_scatter(model, ds, device):
    from torch_geometric.loader import DataLoader

    samples = ds["samples"]
    output_node_idx = ds["output_node_idx"]
    num_nodes = len(ds["all_node_names"])
    output_names = [ds["all_node_names"][i] for i in output_node_idx]

    loader = DataLoader(samples[-300:], batch_size=32)  # held-out-ish slice
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            bsz = batch.num_graphs
            pred_r = pred.view(bsz, num_nodes)[:, output_node_idx]
            true_r = batch.y.view(bsz, num_nodes)[:, output_node_idx]
            all_pred.append(pred_r.cpu().numpy())
            all_true.append(true_r.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(output_names)))
    for i, name in enumerate(output_names):
        ax.scatter(all_true[:, i], all_pred[:, i], s=14, alpha=0.55,
                   color=colors[i], label=name)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("True activation")
    ax.set_ylabel("Predicted activation")
    ax.set_title("Predicted vs. True Output-Layer Activation")
    ax.legend(fontsize=7, loc="upper left", ncol=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "prediction_scatter.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_attention_for_example(model, ds, device, receptor_subset=None):
    """
    Pick one example input pattern, run it through the model with
    return_attention_weights=True, and visualize which edges into the
    final-layer output nodes received the highest attention.
    """
    from pyg_dataset import make_pyg_samples
    from signal_simulator import generate_dataset

    G = ds["graph"]
    all_node_names = ds["all_node_names"]
    receptor_names = ds["receptor_names"]

    if receptor_subset is None:
        receptor_subset = receptor_names[:3]

    # Build one example sample manually
    node_index = {name: i for i, name in enumerate(all_node_names)}
    rng = np.random.default_rng(123)
    _, X, Y, _, _ = generate_dataset(G, num_samples=1, seed=123)
    sample = make_pyg_samples(G, X, Y, receptor_names, all_node_names)[0]
    sample = sample.to(device)

    model.eval()
    with torch.no_grad():
        out, attentions = model(
            sample.x, sample.edge_index, sample.edge_attr,
            return_attention=True,
        )

    # Use the LAST layer's attention (closest to the output predictions)
    edge_idx_att, alpha = attentions[-1]
    alpha_mean = alpha.mean(dim=1).cpu().numpy()  # average over heads
    edge_idx_att = edge_idx_att.cpu().numpy()

    # Build a dataframe-like structure of (src, dst, attn) sorted by attn desc.
    # GATv2Conv adds self-loops internally (each node attending to itself);
    # these get attention weight 1.0 by construction and aren't informative,
    # so we exclude them to focus on real cross-node signaling edges.
    edges_with_attn = []
    for k in range(edge_idx_att.shape[1]):
        src_i, dst_i = edge_idx_att[0, k], edge_idx_att[1, k]
        if src_i == dst_i:
            continue  # skip self-loops
        edges_with_attn.append(
            (all_node_names[src_i], all_node_names[dst_i], float(alpha_mean[k]))
        )
    edges_with_attn.sort(key=lambda t: -t[2])
    top_edges = edges_with_attn[:15]

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [f"{s} -> {d}" for s, d, _ in top_edges]
    values = [a for _, _, a in top_edges]
    ax.barh(range(len(labels)), values, color="#4C72B0")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean attention weight (last GAT layer)")
    active = [r for j, r in enumerate(receptor_names) if X[0, j] == 1.0]
    ax.set_title(
        f"Top Attention Edges for Input Receptors: {', '.join(active)}"
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "attention_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Rebuilding network + dataset for visualization...")
    G = build_signaling_network(seed=42)
    plot_network(G)

    plot_training_curves()

    ds = get_dataset(num_samples=1200, seed=0)
    model = SignalingGNN(num_node_features=ds["num_node_features"]).to(device)
    model_path = os.path.join(
        os.path.dirname(__file__), "models", "best_model.pt"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))

    plot_prediction_scatter(model, ds, device)
    plot_attention_for_example(model, ds, device)

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
