# Signal-GNN: Predicting Downstream Immune Signaling Output from Simultaneous Receptor Activation

A graph neural network that learns to predict the downstream molecular/cytokine
output of an immune signaling network, given which upstream receptors are
simultaneously activated.

## Motivation

This project is a small-scale technical analog of the core ML problem
described by ImmuNet: predicting the molecular output of simultaneous immune
receptor activations, in order to reduce the need for extensive early-stage
experimental screening in drug development.

Rather than using a licensed real-world immunology dataset, this project
builds a **synthetic but structurally realistic** immune signaling network
(receptors → adaptors → kinases → transcription factors → cytokine/effector
outputs, with both activating and inhibitory edges) and simulates signal
propagation through it. This gives a fully self-contained, reproducible
environment for training and evaluating a GNN on exactly the kind of problem
ImmuNet is solving — predicting combinatorial downstream effects from
upstream receptor co-activation — without requiring access to a proprietary
or licensed dataset.

## Problem framing

- **Input:** a binary pattern indicating which receptor(s) are simultaneously
  "active" (e.g. TLR4 + IL1R + RIG-I all firing at once).
- **Graph structure:** a fixed, layered, signed directed graph (48 nodes, ~86
  edges) representing receptors, adaptor proteins, kinases, transcription
  factors, and cytokine/effector outputs — loosely inspired by real innate
  immune signaling biology (e.g. TLR4 → MyD88 → IRAK → NF-kB → TNF-alpha).
- **Output:** the predicted activation level (0–1) of every node in the
  network, with the cytokine/effector **output layer** being the
  biologically meaningful target (analogous to ImmuNet's "molecular output").
- **Model:** a 3-layer Graph Attention Network (GATv2), which passes messages
  along the fixed graph topology and learns how upstream activation patterns
  propagate to downstream nodes — including through inhibitory edges, which
  real immune signaling has plenty of (negative feedback, competing pathways).

## Why GAT specifically

Attention weights provide a built-in interpretability signal: for any given
output prediction, the model exposes which upstream edges it weighted most
heavily. In a drug-discovery context, "which upstream signal contributed most
to this effector molecule's predicted output" is a directly useful question
on top of raw point predictions — see `figures/attention_heatmap.png`, where
the model's top attention weights cleanly correspond to the transcription
factor → cytokine edges that are the most direct causal link to each output
in this network (e.g. `IRF7 -> CXCL10`, `NF-kB -> IL-1beta`), even though
attention was never explicitly trained to recover that structure.

## Results

With 1,200 simulated samples (840 train / 180 val / 180 test) and 60 epochs:

| Metric | Value |
|---|---|
| Test loss (MSE, all nodes) | 0.0003 |
| Test MAE (output-layer nodes only) | 0.018 |

See `figures/training_curves.png` for the full loss curve and
`figures/prediction_scatter.png` for predicted-vs-true activation on every
cytokine/effector output node — predictions track the diagonal closely
across all output molecules.

**Note on the synthetic network:** an earlier version of the network
generator left a few output nodes with zero incoming edges (so they could
never receive any signal) and used an unnormalized weighted sum that caused
high-in-degree nodes to saturate near 1.0 regardless of which specific
receptors were active. Both are fixed in `network_generator.py` (guaranteed
in-degree ≥ 1 for every non-receptor node) and `signal_simulator.py`
(in-degree-normalized input + bias term), so every output node now shows
real, input-dependent variance — see `demo.py` output for a direct
side-by-side of predicted vs. ground-truth activation under a few different
receptor combinations.

## Try it yourself

`demo.py` is the fastest way to see the model work end-to-end without
touching the training pipeline:

```bash
python demo.py --scenario bacterial   # TLR4 + TLR2 + CD14
python demo.py --scenario viral       # RIG-I + MDA5 + TLR7
python demo.py --receptors TLR9 TNFR1 IFNAR1   # any custom combination
python demo.py --list_receptors       # see all valid receptor names
```

Each run prints predicted vs. ground-truth activation for every output
molecule and saves a comparison bar chart to `figures/`.

## Project structure

```
signal-gnn/
├── data/
│   ├── network_generator.py   # builds the layered signaling network
│   ├── signal_simulator.py    # simulates signal propagation -> labels
│   └── pyg_dataset.py         # converts to PyTorch Geometric Data objects
├── models/
│   ├── gnn_model.py            # GATv2-based SignalingGNN architecture
│   ├── train.py                # training loop, early stopping, eval
│   └── best_model.pt           # trained model checkpoint
├── outputs/
│   ├── training_history.json   # per-epoch metrics
│   └── test_metrics.json       # final test-set metrics
├── figures/
│   ├── signaling_network.png   # network topology diagram
│   ├── training_curves.png     # loss / MAE curves
│   ├── prediction_scatter.png  # predicted vs true output activation
│   ├── attention_heatmap.png   # learned attention weights, one example
│   └── demo_*.png              # per-scenario prediction comparisons
├── demo.py                      # standalone inference demo (run this first)
└── visualize.py                 # generates the core training/eval figures
```

## Running it

```bash
pip install torch torch_geometric networkx matplotlib numpy

cd models
python train.py --num_samples 1200 --epochs 60

cd ..
python visualize.py
```

## What I'd explore next (mapping toward ImmuNet's actual problem)

- Swap the synthetic network for real curated pathway data (e.g. Reactome or
  KEGG immune signaling pathways) once licensing/access allows.
- Move from a fixed-topology graph to **inductive** graph learning, so the
  model generalizes to receptor/pathway combinations not seen during
  training, and potentially to entirely new network topologies.
- Add a data-engineering layer to ingest and normalize real expression data
  (e.g. from public immunology datasets) into the same node-feature format
  used here.
- Extend the model to predict not just activation magnitude but also
  *timing/dynamics* of the signaling cascade, since real immune responses are
  temporal, not just steady-state.
