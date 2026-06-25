"""
gnn_model.py

A Graph Attention Network (GAT) that takes a per-sample receptor activation
pattern (encoded as node features on a fixed signaling-network graph) and
predicts the resulting activation level of every node in the network --
most importantly, the output-layer cytokine/effector nodes.

Why GAT (vs. plain GCN)
------------------------
Attention weights give an interpretable signal: for a given prediction, which
upstream neighbors did the model weight most heavily? In a signaling-network
context this is directly useful -- it's a soft, learned analog of asking
"which upstream receptor/kinase contributed most to this cytokine's output,"
which is exactly the kind of explainability a drug-discovery-facing tool like
ImmuNet would want on top of raw predictions.

Architecture
------------
Input node features (6-dim: is_active + 5-dim layer one-hot)
  -> GATConv (multi-head attention, edge-weight-aware)
  -> GATConv
  -> GATConv
  -> Linear output head -> per-node activation prediction (regression, [0,1])

We use 3 GAT layers since the network has up to 4 hops from receptor to
output layer, so 3 layers of message passing lets information from any
receptor reach any output node.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class SignalingGNN(nn.Module):
    def __init__(self, num_node_features: int, hidden_dim: int = 32,
                 num_heads: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(
            GATv2Conv(num_node_features, hidden_dim, heads=num_heads,
                      edge_dim=1, concat=True)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                          edge_dim=1, concat=True)
            )
        # final GAT layer: average heads instead of concatenating, to land
        # on a clean hidden_dim-sized representation before the output head
        self.convs.append(
            GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                      edge_dim=1, concat=False)
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        attentions = []
        h = x
        for i, conv in enumerate(self.convs):
            if return_attention:
                h, (edge_idx_att, alpha) = conv(
                    h, edge_index, edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                attentions.append((edge_idx_att, alpha))
            else:
                h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.output_head(h).squeeze(-1)  # (num_nodes,)
        out = torch.sigmoid(out)  # activations are in [0, 1]

        if return_attention:
            return out, attentions
        return out


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from pyg_dataset import get_dataset
    from torch_geometric.loader import DataLoader

    ds = get_dataset(num_samples=4)
    loader = DataLoader(ds["samples"], batch_size=2)
    model = SignalingGNN(num_node_features=ds["num_node_features"])

    batch = next(iter(loader))
    out = model(batch.x, batch.edge_index, batch.edge_attr)
    print("Output shape:", out.shape)  # (batch_size * num_nodes,)
    print("Sample outputs:", out[:10])
