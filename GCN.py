from pathlib import Path
from collections import defaultdict
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as W
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv

# point to folder where the graphs live (created after adjusting cosine)
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# find the graph files
paths = sorted(PROJECT_ROOT.glob("review_similarity_graph_*.pt"))

print("cosine", "num nodes", "num_edges")
# loop over to get the files
for p in paths:
	# load graph
	graph = torch.load(p, weights_only=False)
	# get the 0.75 , 0.9 etc
	clean_name = p.stem
	match = re.search(r"\d+", clean_name)
	if match:
		number_part = match.group()
	print(number_part, graph.num_nodes, graph.num_edges)




# Splitting the data using a random seed
# this makes sure we get the same split every time
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

num_reviews = graph.num_nodes
labels = graph.y.numpy()  # the label we are trying to predict
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


# second loop: attach the same masks to every threshold graph
for p in paths:
	graph = torch.load(p, weights_only=False)
	graph.train_mask = train_mask
	graph.val_mask = val_mask
	graph.test_mask = test_mask
	print(p.stem, train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())


class GCNBaseline(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
		super().__init__()
		self.conv1 = GCNConv(in_channels, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, out_channels)
		self.dropout = dropout

	def forward(self, x, edge_index):
		x = W.relu(self.conv1(x, edge_index))
		x = W.dropout(x, p=self.dropout, training=self.training)
		return self.conv2(x, edge_index)


def _metrics(y_true, y_pred):
	return {
		"Accuracy": accuracy_score(y_true, y_pred),
		"Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
		"Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
		"Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
	}


def train_and_evaluate(model_name, model, data, epochs=100, lr=0.01, patience=20):
	"""Train a Graph-1 GCN. Uses train_mask for loss, val_mask to pick best epoch, test_mask for final score."""
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


# --- Train GCN on threshold graphs (same masks each time) ---
# Skip 0.60 for now (too many edges / too slow). Original baseline was 0.75.
train_paths = [
	p for p in paths
	if re.search(r"_(\d+)\.pt$", p.name)
	and int(re.search(r"_(\d+)\.pt$", p.name).group(1)) in {75, 80, 90}
]

print(f"\nDevice: {DEVICE}")
print("Training thresholds:", [p.name for p in train_paths])
all_results = []

for p in train_paths:
	data = torch.load(p, weights_only=False)
	data.train_mask = train_mask
	data.val_mask = val_mask
	data.test_mask = test_mask

	# threshold from filename, e.g. review_similarity_graph_75 -> 0.75
	match = re.search(r"\d+", p.stem)
	threshold = int(match.group()) / 100 if match else None

	in_dim = data.num_features
	num_classes = int(torch.unique(data.y).numel())
	# new model each threshold so runs are independent
	model = GCNBaseline(in_dim, 64, num_classes)

	print(f"\nTraining on: {p.name} | threshold={threshold} | edges={data.num_edges}")
	metrics, history = train_and_evaluate(f"GCN_t{threshold}", model, data)

	row = {
		"threshold": threshold,
		"edges": int(data.num_edges),
		**metrics,
	}
	all_results.append(row)

print("\n========== Threshold sweep summary (GCN) ==========")
print(f"{'t':>6} | {'edges':>10} | {'Acc':>8} | {'Macro F1':>8} | {'Val Acc':>8}")
print("-" * 55)
for row in all_results:
	print(
		f"{row['threshold']:>6} | {row['edges']:>10} | "
		f"{row['Accuracy']:>8.4f} | {row['Macro F1']:>8.4f} | "
		f"{row['Best Val Acc']:>8.4f}"
	)

# save for later comparison / slides
results_path = PROJECT_ROOT / "threshold_gcn_results.csv"
with results_path.open("w", newline="") as f:
	import csv

	writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
	writer.writeheader()
	writer.writerows(all_results)
print(f"\nSaved results to {results_path}")

# --- Train GCN on AND graphs (text + rating, d=2) ---
and_paths = sorted(PROJECT_ROOT.glob("review_similarity_graph_and_d1_*.pt"))
and_paths = [
    p for p in and_paths
    if int(re.search(r"(\d+)$", p.stem).group(1)) in {75, 80, 90}
]

print(f"\nDevice: {DEVICE}")
print("Training AND graphs:", [p.name for p in and_paths])
if not and_paths:
	raise FileNotFoundError(
		"No review_similarity_graph_and_d1_*.pt files found. "
		"Run python3 main.py first to build the d=1 AND graphs."
	)
all_results = []

for p in and_paths:
    data = torch.load(p, weights_only=False)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    match = re.search(r"(\d+)$", p.stem)
    threshold = int(match.group(1)) / 100 if match else None

    in_dim = data.num_features
    num_classes = int(torch.unique(data.y).numel())
    model = GCNBaseline(in_dim, 64, num_classes)

    print(f"\nTraining on: {p.name} | threshold={threshold} | edges={data.num_edges}")
    metrics, history = train_and_evaluate(f"GCN_AND_d1_t{threshold}", model, data)

    row = {
        "method": "and_d1",
        "threshold": threshold,
        "edges": int(data.num_edges),
        **metrics,
    }
    all_results.append(row)

print("\n========== AND (text+rating d=1) summary (GCN) ==========")
print(f"{'t':>6} | {'edges'clear:>10} | {'Acc':>8} | {'Macro F1':>8} | {'Val Acc':>8}")
print("-" * 55)
for row in all_results:
    print(
        f"{row['threshold']:>6} | {row['edges']:>10} | "
        f"{row['Accuracy']:>8.4f} | {row['Macro F1']:>8.4f} | "
        f"{row['Best Val Acc']:>8.4f}"
    )

import csv
results_path = PROJECT_ROOT / "threshold_gcn_and_d1_results.csv"
with results_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
    writer.writeheader()
    writer.writerows(all_results)
print(f"\nSaved results to {results_path}")
