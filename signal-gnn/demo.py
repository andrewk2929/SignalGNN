"""
demo.py

A clean, standalone inference demo: pick which receptors are "active" and
immediately see the model's predicted downstream cytokine/effector output.

This is the file to run if you just want to see the model *work*, without
touching the training pipeline.

Usage
-----
    # Try a couple of built-in example scenarios:
    python demo.py --scenario bacterial
    python demo.py --scenario viral
    python demo.py --scenario inflammasome

    # Or specify your own receptor combination:
    python demo.py --receptors TLR4 IL1R TNFR1

    # List all valid receptor names:
    python demo.py --list_receptors

Each run prints a readable table of predicted output-layer (cytokine/effector)
activations, compares it against the ground-truth simulator (since we know
the "true" propagation rule here), and saves a bar-chart figure.
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from network_generator import build_signaling_network, get_layer_nodes, LAYER_ORDER
from signal_simulator import simulate_one
from pyg_dataset import build_static_graph_tensors
from gnn_model import SignalingGNN
from torch_geometric.data import Data


# A few illustrative scenarios mirroring real innate-immune activation
# contexts (named for readability; receptor choices are simplified, not a
# literal clinical claim).
SCENARIOS = {
    "bacterial": ["TLR4", "TLR2", "CD14"],       # gram-negative/positive cell wall sensing
    "viral": ["RIG-I", "MDA5", "TLR7"],          # viral RNA sensing
    "sterile_inflammation": ["NLRP3", "IL1R"],   # inflammasome / damage-associated signaling
    "mixed_infection": ["TLR4", "RIG-I", "IL1R", "TLR9"],
}


def load_model_and_graph():
    G = build_signaling_network(seed=42)
    all_node_names = list(G.nodes())
    receptor_names = get_layer_nodes(G, "receptor")
    output_names = get_layer_nodes(G, "output")
    node_index = {name: i for i, name in enumerate(all_node_names)}

    edge_index, edge_weight, layer_feat = build_static_graph_tensors(
        G, all_node_names
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_node_features = layer_feat.shape[1] + 1  # +1 for is_active flag
    model = SignalingGNN(num_node_features=num_node_features).to(device)
    model_path = os.path.join(
        os.path.dirname(__file__), "models", "best_model.pt"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return {
        "G": G,
        "all_node_names": all_node_names,
        "receptor_names": receptor_names,
        "output_names": output_names,
        "node_index": node_index,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "layer_feat": layer_feat,
        "model": model,
        "device": device,
    }


def predict(ctx, active_receptors):
    """Run the GNN on a chosen set of active receptors, return predicted
    activation for every node (as a name -> value dict)."""
    num_nodes = len(ctx["all_node_names"])
    is_active = torch.zeros((num_nodes, 1), dtype=torch.float32)
    for r in active_receptors:
        is_active[ctx["node_index"][r], 0] = 1.0

    x = torch.cat([is_active, ctx["layer_feat"]], dim=1).to(ctx["device"])
    edge_index = ctx["edge_index"].to(ctx["device"])
    edge_attr = ctx["edge_weight"].unsqueeze(1).to(ctx["device"])

    with torch.no_grad():
        out = ctx["model"](x, edge_index, edge_attr)

    out = out.cpu().numpy()
    return {name: float(out[i]) for i, name in enumerate(ctx["all_node_names"])}


def ground_truth(ctx, active_receptors, seed=0):
    """Run the original hand-written simulator (the 'ground truth' rule the
    GNN was trained to approximate) for comparison."""
    rng = np.random.default_rng(seed)
    activation = simulate_one(
        ctx["G"], ctx["node_index"], set(active_receptors), rng, noise_std=0.0
    )
    return {name: float(activation[i]) for i, name in enumerate(ctx["all_node_names"])}


def print_table(ctx, active_receptors, pred, true):
    print("\n" + "=" * 64)
    print(f"ACTIVE RECEPTORS: {', '.join(active_receptors)}")
    print("=" * 64)
    print(f"\n{'OUTPUT MOLECULE':<16}{'PREDICTED':>12}{'GROUND TRUTH':>16}{'DIFF':>10}")
    print("-" * 54)
    for name in ctx["output_names"]:
        p, t = pred[name], true[name]
        print(f"{name:<16}{p:>12.3f}{t:>16.3f}{abs(p - t):>10.3f}")
    print()


def plot_comparison(ctx, active_receptors, pred, true, out_path):
    names = ctx["output_names"]
    pred_vals = [pred[n] for n in names]
    true_vals = [true[n] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, true_vals, width, label="Ground truth (simulator)",
           color="#999999")
    ax.bar(x + width / 2, pred_vals, width, label="GNN prediction",
           color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Activation level")
    ax.set_title(f"Predicted vs. Ground-Truth Output\n"
                 f"Active receptors: {', '.join(active_receptors)}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptors", nargs="+", default=None,
                         help="Receptor names to activate, e.g. TLR4 IL1R")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                         default=None, help="Use a built-in example scenario")
    parser.add_argument("--list_receptors", action="store_true",
                         help="List valid receptor names and exit")
    parser.add_argument("--out", default=None,
                         help="Output path for comparison chart "
                              "(default: figures/demo_<scenario>.png)")
    args = parser.parse_args()

    ctx = load_model_and_graph()

    if args.list_receptors:
        print("Valid receptor names:")
        for r in ctx["receptor_names"]:
            print(f"  {r}")
        return

    if args.receptors:
        active_receptors = args.receptors
        scenario_tag = "custom"
    elif args.scenario:
        active_receptors = SCENARIOS[args.scenario]
        scenario_tag = args.scenario
    else:
        active_receptors = SCENARIOS["bacterial"]
        scenario_tag = "bacterial"
        print("(No --receptors or --scenario given, defaulting to "
              "'bacterial' example. Use --list_receptors to see all options.)")

    invalid = [r for r in active_receptors if r not in ctx["receptor_names"]]
    if invalid:
        print(f"ERROR: unknown receptor name(s): {invalid}")
        print("Run with --list_receptors to see valid names.")
        sys.exit(1)

    pred = predict(ctx, active_receptors)
    true = ground_truth(ctx, active_receptors)

    print_table(ctx, active_receptors, pred, true)

    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "figures", f"demo_{scenario_tag}.png"
    )
    plot_comparison(ctx, active_receptors, pred, true, out_path)


if __name__ == "__main__":
    main()
