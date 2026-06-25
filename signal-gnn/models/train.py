"""
train.py

Trains the SignalingGNN to predict full-network activation states from
receptor activation patterns, then evaluates specifically on the
output-layer (cytokine/effector) nodes, since that's the biologically
meaningful prediction target (analogous to ImmuNet predicting molecular
output from simultaneous receptor activation).

Usage
-----
    python train.py --epochs 150 --num_samples 3000

Outputs
-------
    models/best_model.pt          -- best model checkpoint (by val loss)
    outputs/training_history.json -- per-epoch train/val loss + output-node MAE
    outputs/test_metrics.json      -- final test-set metrics
"""

import argparse
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from pyg_dataset import get_dataset
from gnn_model import SignalingGNN


def split_samples(samples, train_frac=0.7, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)

    n_train = int(len(samples) * train_frac)
    n_val = int(len(samples) * val_frac)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    test = [samples[i] for i in test_idx]
    return train, val, test


def evaluate(model, loader, output_node_idx, num_nodes, device, criterion):
    model.eval()
    total_loss = 0.0
    total_output_abs_err = 0.0
    total_output_count = 0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(pred, batch.y)
            total_loss += loss.item()
            n_batches += 1

            # Reshape to (batch_size, num_nodes) to slice out output nodes
            bsz = batch.num_graphs
            pred_r = pred.view(bsz, num_nodes)
            y_r = batch.y.view(bsz, num_nodes)

            out_pred = pred_r[:, output_node_idx]
            out_true = y_r[:, output_node_idx]
            total_output_abs_err += torch.abs(out_pred - out_true).sum().item()
            total_output_count += out_pred.numel()

    avg_loss = total_loss / max(n_batches, 1)
    output_mae = total_output_abs_err / max(total_output_count, 1)
    return avg_loss, output_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=20,
                         help="early stopping patience (epochs without "
                              "val improvement)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Building dataset...")
    ds = get_dataset(num_samples=args.num_samples, seed=args.seed)
    samples = ds["samples"]
    num_nodes = len(ds["all_node_names"])
    output_node_idx = ds["output_node_idx"]

    train_samples, val_samples, test_samples = split_samples(
        samples, seed=args.seed
    )
    print(f"Train/Val/Test sizes: "
          f"{len(train_samples)}/{len(val_samples)}/{len(test_samples)}")

    train_loader = DataLoader(train_samples, batch_size=args.batch_size,
                               shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=args.batch_size)
    test_loader = DataLoader(test_samples, batch_size=args.batch_size)

    model = SignalingGNN(
        num_node_features=ds["num_node_features"],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    os.makedirs("models", exist_ok=True) if os.path.basename(
        os.getcwd()) != "models" else None
    model_dir = os.path.join(os.path.dirname(__file__))
    best_model_path = os.path.join(model_dir, "best_model.pt")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)
        val_loss, val_output_mae = evaluate(
            model, val_loader, output_node_idx, num_nodes, device, criterion
        )
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_output_mae": val_output_mae,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_output_MAE={val_output_mae:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(best epoch was {best_epoch})")
                break

    print(f"\nLoading best model from epoch {best_epoch} "
          f"(val_loss={best_val_loss:.4f})")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_output_mae = evaluate(
        model, test_loader, output_node_idx, num_nodes, device, criterion
    )
    print(f"Test loss (MSE, all nodes): {test_loss:.4f}")
    print(f"Test MAE (output-layer nodes only): {test_output_mae:.4f}")

    os.makedirs("../outputs", exist_ok=True)
    with open("../outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open("../outputs/test_metrics.json", "w") as f:
        json.dump({
            "test_loss_mse_all_nodes": test_loss,
            "test_mae_output_nodes": test_output_mae,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "num_samples": args.num_samples,
            "train_size": len(train_samples),
            "val_size": len(val_samples),
            "test_size": len(test_samples),
        }, f, indent=2)

    print("\nSaved training history -> outputs/training_history.json")
    print("Saved test metrics -> outputs/test_metrics.json")
    print(f"Saved best model -> models/best_model.pt")


if __name__ == "__main__":
    main()
