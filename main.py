import os
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from torch_geometric.data import Data, HeteroData
from tqdm import tqdm

N_JOBS = max(1, (os.cpu_count() or 2) - 1)
TEXT_CHUNK = 250
SIM_CHUNK = 1024


def _torch_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _chunked(items, size):
    return [items[i : i + size] for i in range(0, len(items), size)]


def clean_currency_series(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan}),
        errors="coerce",
    )


def calculate_tier(group):
    if group["price_per_sqft"].dropna().nunique() < 2:
        group["market_tier"] = 1
        return group

    q25 = group["price_per_sqft"].quantile(0.25)
    q75 = group["price_per_sqft"].quantile(0.75)
    conditions = [
        (group["price_per_sqft"] <= q25),
        (group["price_per_sqft"] > q25) & (group["price_per_sqft"] <= q75),
        (group["price_per_sqft"] > q75),
    ]
    group["market_tier"] = np.select(conditions, [0, 1, 2], default=np.nan)
    return group


def calc_lexical_diversity(text):
    words = str(text).lower().split()
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)


def get_imperative_density(text):
    blob = TextBlob(str(text))
    sentences = blob.sentences
    if len(sentences) == 0:
        return 0.0

    imperative_count = 0
    for sentence in sentences:
        tags = sentence.tags
        if tags and tags[0][1] == "VB":
            imperative_count += 1
    return imperative_count / len(sentences)


def _sentiment_chunk(texts):
    polarities = []
    subjectivities = []
    for text in texts:
        sentiment = TextBlob(str(text)).sentiment
        polarities.append(sentiment.polarity)
        subjectivities.append(sentiment.subjectivity)
    return polarities, subjectivities


def _imperative_chunk(texts):
    return [get_imperative_density(text) for text in texts]


def _diversity_chunk(texts):
    return [calc_lexical_diversity(text) for text in texts]


def _parallel_chunks(fn, items, desc):
    chunks = _chunked(items, TEXT_CHUNK)
    parts = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(fn)(chunk) for chunk in tqdm(chunks, desc=desc)
    )
    return parts


def cosine_similarity_edges(features, threshold=0.75, chunk_size=SIM_CHUNK, device=None):
    """Same rule as sklearn radius_neighbors_graph: cosine sim >= threshold, no self-loops."""
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    x = F.normalize(x, p=2, dim=1)
    n = x.size(0)
    sources = []
    targets = []
    for start in tqdm(range(0, n, chunk_size), desc="Similarity edges"):
        end = min(start + chunk_size, n)
        sims = x[start:end] @ x.T
        hits = (sims >= threshold).nonzero(as_tuple=False)
        i = hits[:, 0] + start
        j = hits[:, 1]
        keep = i != j
        sources.append(i[keep].detach().cpu())
        targets.append(j[keep].detach().cpu())
        del sims
    return torch.stack((torch.cat(sources), torch.cat(targets)), dim=0)


def main():
    print("--- Step 1: Loading dataset ---")
    df = pd.read_csv("kmeans_binary_labeled_helpfulness_data_final.csv")
    df["review"] = df["review"].fillna("")

    df["minRent_num"] = clean_currency_series(df["minRent"])
    df["maxRent_num"] = clean_currency_series(df["maxRent"])
    df["rent_mid"] = (df["minRent_num"] + df["maxRent_num"]) / 2
    df["sqft_mid"] = (df["minSqft"] + df["maxSqft"]) / 2
    df["price_per_sqft"] = np.where(df["sqft_mid"] > 0, df["rent_mid"] / df["sqft_mid"], np.nan)

    df = df.groupby("zip", group_keys=False).apply(calculate_tier)
    df = df.reset_index(drop=True)

    df["review_length"] = df["review"].str.len()
    df["rating_deviation"] = abs(df["rating"] - df["avgRating"])

    reviews = df["review"].tolist()

    print(f"Calculating sentiment + subjectivity on {N_JOBS} cores (one TextBlob pass)...")
    sent_parts = _parallel_chunks(_sentiment_chunk, reviews, "Sentiment/subjectivity")
    polarities, subjectivities = [], []
    for p, s in sent_parts:
        polarities.extend(p)
        subjectivities.extend(s)
    df["sentiment_score"] = polarities
    df["subjectivity_score"] = subjectivities

    device = _torch_device()
    print(f"Encoding reviews with Sentence-BERT on {device}...")
    sbert_device = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", device=sbert_device)
        embeddings = model.encode(
            reviews,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    except Exception as exc:
        print(f"GPU encode failed ({exc}); retrying on CPU.")
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        embeddings = model.encode(
            reviews,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        device = torch.device("cpu")
    df["sbert_embedding"] = list(embeddings)

    print("\n--- Sentiment vs. Subjectivity Sample ---")
    print(df[["review", "sentiment_score", "subjectivity_score"]].head())
    print("\n--- Feature Engineering Complete ---")
    print(df[["rating_deviation", "review_length", "sentiment_score", "market_tier"]].head())
    print(f"Embedding Array Shape per Node: {embeddings.shape[1]} dimensions")

    print("Calculating Lexical Diversity (Type-Token Ratio)...")
    div_parts = _parallel_chunks(_diversity_chunk, reviews, "Lexical diversity")
    df["lexical_diversity"] = [v for part in div_parts for v in part]
    print("\n--- Lexical Diversity Sample ---")
    print(df[["review_length", "lexical_diversity"]].head())

    print("Calculating Objective Imperative Density (Actionability)...")
    imp_parts = _parallel_chunks(_imperative_chunk, reviews, "Imperative density")
    df["imperative_density"] = [v for part in imp_parts for v in part]
    print("\n--- Imperative Density Sample ---")
    print(df[["review_length", "imperative_density"]].head())

    print("\n--- Constructing the Review Similarity Graph Edges ---")
    scalar_features = df[
        [
            "review_length",
            "rating_deviation",
            "sentiment_score",
            "market_tier",
            "subjectivity_score",
            "lexical_diversity",
            "imperative_density",
        ]
    ].fillna(0).values
    sbert_features = np.vstack(df["sbert_embedding"].values)
    node_features = np.hstack((scalar_features, sbert_features))
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(df["kmeans_label"].values, dtype=torch.long)

    print("Calculating edges (cosine similarity >= 0.75, chunked)...")
    edge_index = cosine_similarity_edges(sbert_features, threshold=0.75, device=device)

    graph_data = Data(x=x, edge_index=edge_index, y=y)
    print("\n--- Final Graph Construction Complete ---")
    print(f"Number of Nodes: {graph_data.num_nodes}")
    print(f"Number of Edges: {graph_data.num_edges}")
    print(f"Node Feature Dimension: {graph_data.num_features}")
    print(f"Target Labels Shape: {graph_data.y.shape}")
    torch.save(graph_data, "review_similarity_graph.pt")
    print("Graph successfully saved to 'review_similarity_graph.pt'")

    print("\n--- Constructing Task 2: Bipartite Graph (Review-to-Apartment) ---")
    apt_mapping = {url: i for i, url in enumerate(df["url"].unique())}
    df["apt_id"] = df["url"].map(apt_mapping)
    num_reviews = len(df)
    num_apartments = len(apt_mapping)
    print(f"Found {num_reviews} Reviews belonging to {num_apartments} unique Apartments.")

    apt_df = df.groupby("apt_id").agg(
        {
            "avgRating": "first",
            "rent_mid": "first",
            "sqft_mid": "first",
            "market_tier": "first",
        }
    ).fillna(0)
    apartment_features = torch.tensor(apt_df.values, dtype=torch.float)

    review_features = x.clone()
    review_labels = y.clone()
    review_indices = np.arange(len(df))
    apartment_indices = df["apt_id"].values
    edge_index_rev_to_apt = torch.tensor(
        np.vstack((review_indices, apartment_indices)), dtype=torch.long
    )
    edge_index_apt_to_rev = torch.tensor(
        np.vstack((apartment_indices, review_indices)), dtype=torch.long
    )

    bipartite_graph = HeteroData()
    bipartite_graph["review"].x = review_features
    bipartite_graph["review"].y = review_labels
    bipartite_graph["apartment"].x = apartment_features
    bipartite_graph["review", "belongs_to", "apartment"].edge_index = edge_index_rev_to_apt
    bipartite_graph["apartment", "has_review", "review"].edge_index = edge_index_apt_to_rev

    print("\n--- Bipartite Graph Construction Complete ---")
    print(bipartite_graph)
    torch.save(bipartite_graph, "bipartite_review_apartment_graph.pt")
    print("Task 2 Graph successfully saved to 'bipartite_review_apartment_graph.pt'")


if __name__ == "__main__":
    main()
