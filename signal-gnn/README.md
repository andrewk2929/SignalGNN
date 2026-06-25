# Signal-GNN: Predicting Downstream Immune Signaling Output from Simultaneous Receptor Activation

A graph neural network that learns to predict the downstream molecular/cytokine
output of an immune signaling network, given which upstream receptors are
simultaneously activated.

## Problem framing

- **Input:** a binary pattern indicating which receptor(s) are simultaneously
  "active" (e.g. TLR4 + IL1R + RIG-I all firing at once).
- **Graph structure:** a fixed, layered, signed directed graph (48 nodes, ~86
  edges) representing receptors, adaptor proteins, kinases, transcription
  factors, and cytokine/effector outputs — loosely inspired by real innate
  immune signaling biology (e.g. TLR4 → MyD88 → IRAK → NF-kB → TNF-alpha).
- **Output:** the predicted activation level (0–1) of every node in the
  network, with the cytokine/effector **output layer** being the
  biologically meaningful target.
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

