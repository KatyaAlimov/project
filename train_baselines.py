"""
Baseline GNN evaluation on the review similarity graph (Graph 1)
and the review–apartment bipartite graph (Graph 2).

Bug this script repairs
-----------------------
Graph 2 was built with `df.index.values` after a `groupby` that dropped
NaN-zip rows and left gaps in the pandas index. Edge endpoints therefore
did not match positional node IDs in `x`, so most edges were either
invalid or pointed at the wrong reviews. We rebuild those edges here with
`np.arange(len(df))` after `reset_index(drop=True)`.
"""

import csv
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GCNConv, HeteroConv, SAGEConv

print("--- Step 1: Loading graphs, repairing Graph 2 edges, building splits ---")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Load graphs
# ---------------------------------------------------------------------------
graph_1 = torch.load("review_similarity_graph.pt", weights_only=False)
graph_2 = torch.load("bipartite_review_apartment_graph.pt", weights_only=False)

num_reviews = graph_1.num_nodes
labels = graph_1.y.numpy()


def rebuild_bipartite_edges(hetero_graph: HeteroData) -> HeteroData:
    """Recompute review↔apartment edges with positional node IDs."""
    df = pd.read_csv("kmeans_binary_labeled_helpfulness_data_final.csv")

    # Same row filter as main.py: groupby('zip') drops NaN-zip rows.
    df = df[df["zip"].notna()].copy().reset_index(drop=True)

    if len(df) != hetero_graph["review"].num_nodes:
        raise ValueError(
            f"CSV alignment failed: {len(df)} rows vs "
            f"{hetero_graph['review'].num_nodes} review nodes"
        )

    y_check = torch.tensor(df["kmeans_label"].values, dtype=torch.long)
    if not torch.equal(y_check, hetero_graph["review"].y.cpu()):
        raise ValueError("Label order from CSV does not match saved review.y")

    apt_mapping = {url: i for i, url in enumerate(df["url"].unique())}
    df["apt_id"] = df["url"].map(apt_mapping)

    review_indices = np.arange(len(df), dtype=np.int64)
    apartment_indices = df["apt_id"].to_numpy(dtype=np.int64)

    if apartment_indices.max() >= hetero_graph["apartment"].num_nodes:
        raise ValueError("Apartment id exceeds apartment feature matrix size")

    edge_rev_to_apt = torch.tensor(
        np.vstack((review_indices, apartment_indices)), dtype=torch.long
    )
    edge_apt_to_rev = edge_rev_to_apt.flip(0)

    hetero_graph["review", "belongs_to", "apartment"].edge_index = edge_rev_to_apt
    hetero_graph["apartment", "has_review", "review"].edge_index = edge_apt_to_rev

    print(
        f"Repaired Graph 2 edges: {edge_rev_to_apt.size(1)} "
        f"(expected ~{num_reviews}, one per review)"
    )
    return hetero_graph


graph_2 = rebuild_bipartite_edges(graph_2)

# Persist repaired edges only (raw features). Normalization is applied
# in-memory below so re-running this script does not double-scale x.
torch.save(graph_2, "bipartite_review_apartment_graph.pt")

# ---------------------------------------------------------------------------
# Stratified 70 / 15 / 15 split (shared across all models)
# ---------------------------------------------------------------------------
indices = np.arange(num_reviews)
train_idx, temp_idx = train_test_split(
    indices, test_size=0.30, random_state=SEED, stratify=labels
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=SEED, stratify=labels[temp_idx]
)

train_mask = torch.zeros(num_reviews, dtype=torch.bool)
val_mask = torch.zeros(num_reviews, dtype=torch.bool)
test_mask = torch.zeros(num_reviews, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

graph_1.train_mask = train_mask
graph_1.val_mask = val_mask
graph_1.test_mask = test_mask
graph_2["review"].train_mask = train_mask.clone()
graph_2["review"].val_mask = val_mask.clone()
graph_2["review"].test_mask = test_mask.clone()

torch.save(graph_1, "review_similarity_graph.pt")


def standardize_features_(x: torch.Tensor, mask: torch.Tensor) -> None:
    """In-place z-score using train rows only (avoids leakage)."""
    mean = x[mask].mean(dim=0)
    std = x[mask].std(dim=0).clamp_min(1e-6)
    x.sub_(mean).div_(std)


# Working copies for training (do not write normalized features back to disk)
train_graph_1 = graph_1.clone()
train_graph_2 = graph_2.clone()
standardize_features_(train_graph_1.x, train_mask)
standardize_features_(train_graph_2["review"].x, train_mask)
standardize_features_(
    train_graph_2["apartment"].x,
    torch.ones(train_graph_2["apartment"].num_nodes, dtype=torch.bool),
)

print(
    f"Masks: train={train_mask.sum().item()} | "
    f"val={val_mask.sum().item()} | test={test_mask.sum().item()}"
)


# =============================================================
# Models
# =============================================================

class MLPBaseline(nn.Module):
    """Feature-only control: no message passing (structure ablation)."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class GCNBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGEBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GATBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class HeteroSAGEBaseline(nn.Module):
    def __init__(self, review_in_dim, apt_in_dim, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.proj_review = nn.Linear(review_in_dim, hidden_channels)
        self.proj_apt = nn.Linear(apt_in_dim, hidden_channels)
        self.conv1 = HeteroConv(
            {
                ("review", "belongs_to", "apartment"): SAGEConv(hidden_channels, hidden_channels),
                ("apartment", "has_review", "review"): SAGEConv(hidden_channels, hidden_channels),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("review", "belongs_to", "apartment"): SAGEConv(hidden_channels, hidden_channels),
                ("apartment", "has_review", "review"): SAGEConv(hidden_channels, out_channels),
            },
            aggr="sum",
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            "review": self.proj_review(x_dict["review"]),
            "apartment": self.proj_apt(x_dict["apartment"]),
        }
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {
            k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()
        }
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict["review"]


# =============================================================
# Train / eval
# =============================================================

def _metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_and_evaluate(model_name, model, data, epochs=100, lr=0.01, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    data = data.to(DEVICE)

    best_val_acc = -1.0
    best_state = None
    best_test_metrics = {}
    history = defaultdict(list)
    stalled = 0

    print(f"\n================ Training {model_name} on Graph 1 ================")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=-1)
            val_acc = accuracy_score(
                data.y[data.val_mask].cpu(), preds[data.val_mask].cpu()
            )

        history["loss"].append(float(loss.item()))
        history["val_acc"].append(float(val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stalled = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = preds[data.test_mask].cpu().numpy()
            best_test_metrics = _metrics(y_true, y_pred)
            best_test_metrics["Best Val Acc"] = best_val_acc
            best_test_metrics["Best Epoch"] = epoch
        else:
            stalled += 1

        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

        if stalled >= patience:
            print(f"Early stop at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n--- Best Test Results for {model_name} ---")
    for metric, value in best_test_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    return best_test_metrics, history


def train_and_evaluate_hetero(model_name, model, data, epochs=100, lr=0.01, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    data = data.to(DEVICE)

    best_val_acc = -1.0
    best_state = None
    best_test_metrics = {}
    history = defaultdict(list)
    stalled = 0

    print(f"\n================ Training {model_name} on Graph 2 (Bipartite) ================")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = criterion(
            out[data["review"].train_mask],
            data["review"].y[data["review"].train_mask],
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
            preds = out.argmax(dim=-1)
            val_acc = accuracy_score(
                data["review"].y[data["review"].val_mask].cpu(),
                preds[data["review"].val_mask].cpu(),
            )

        history["loss"].append(float(loss.item()))
        history["val_acc"].append(float(val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stalled = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            y_true = data["review"].y[data["review"].test_mask].cpu().numpy()
            y_pred = preds[data["review"].test_mask].cpu().numpy()
            best_test_metrics = _metrics(y_true, y_pred)
            best_test_metrics["Best Val Acc"] = best_val_acc
            best_test_metrics["Best Epoch"] = epoch
        else:
            stalled += 1

        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

        if stalled >= patience:
            print(f"Early stop at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n--- Best Test Results for {model_name} on Graph 2 ---")
    for metric, value in best_test_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    return best_test_metrics, history


# =============================================================
# Execution
# =============================================================

in_dim = train_graph_1.num_features
num_classes = int(torch.unique(train_graph_1.y).numel())

homo_models = {
    "MLP (no graph)": MLPBaseline(in_dim, 64, num_classes),
    "GCN": GCNBaseline(in_dim, 64, num_classes),
    "GraphSAGE": GraphSAGEBaseline(in_dim, 64, num_classes),
    "GAT": GATBaseline(in_dim, 64, num_classes),
}

results = {}
histories = {}
for name, model in homo_models.items():
    data = train_graph_1.clone()
    metrics, hist = train_and_evaluate(name, model, data)
    results[f"{name} | Graph1"] = metrics
    histories[f"{name} | Graph1"] = hist

review_dim = train_graph_2["review"].x.shape[1]
apt_dim = train_graph_2["apartment"].x.shape[1]
hetero = HeteroSAGEBaseline(review_dim, apt_dim, 64, num_classes)
metrics, hist = train_and_evaluate_hetero("HeteroSAGE", hetero, train_graph_2.clone())
results["HeteroSAGE | Graph2"] = metrics
histories["HeteroSAGE | Graph2"] = hist

# ---------------------------------------------------------------------------
# Comparison table + CSV
# ---------------------------------------------------------------------------
metric_cols = ["Accuracy", "Macro F1", "Precision", "Recall", "Best Val Acc", "Best Epoch"]
print("\n" + "=" * 78)
print(f"{'Model':<28} " + " ".join(f"{c:>11}" for c in metric_cols[:-1]))
print("-" * 78)
for name, m in results.items():
    row = " ".join(
        f"{m[c]:11.4f}" if isinstance(m[c], float) else f"{m[c]:>11}"
        for c in metric_cols[:-1]
    )
    print(f"{name:<28} {row}")
print("=" * 78)

with open("baseline_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Model", *metric_cols])
    writer.writeheader()
    for name, m in results.items():
        writer.writerow({"Model": name, **{c: m[c] for c in metric_cols}})

# Epoch histories for plotting later
with open("baseline_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Epoch", "Loss", "Val Acc"])
    for name, hist in histories.items():
        for i, (loss, val_acc) in enumerate(zip(hist["loss"], hist["val_acc"]), start=1):
            writer.writerow([name, i, loss, val_acc])

print("\nSaved baseline_results.csv and baseline_history.csv")
