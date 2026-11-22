#!/usr/bin/env python3
"""Build DiSC_Project_Presentation.pptx from the DiSC Lab template."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Pt

PROJECT = Path(__file__).resolve().parent
TEMPLATE = PROJECT / "DiSC-Lab-Template-ICPP.pptx"
OUTPUT = PROJECT / "DiSC_Project_Presentation.pptx"

TITLE = "Predicting Apartment Review Helpfulness with Graph Neural Networks"


def delete_all_slides(prs: Presentation) -> None:
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        rId = sldId.get(qn("r:id"))
        prs.part.drop_rel(rId)
        sldIdLst.remove(sldId)


def set_runs(paragraph, texts: list[str]) -> None:
    """Replace paragraph runs with texts, cloning the first run's formatting."""
    if not paragraph.runs:
        paragraph.text = texts[0] if texts else ""
        return
    first = paragraph.runs[0]
    # Keep first run, clear others
    for r in paragraph.runs[1:]:
        r._r.getparent().remove(r._r)
    first.text = texts[0] if texts else ""
    for extra in texts[1:]:
        new_r = copy.deepcopy(first._r)
        # clear text nodes then set
        for child in list(new_r):
            if child.tag == qn("a:t"):
                child.text = extra
        first._r.addnext(new_r)
        first = paragraph.runs[-1]


def clear_and_bullets(tf, items: list[tuple[int, str]], *, preserve_first_style: bool = True) -> None:
    """
    items: list of (level, text).
    Reuses existing paragraph XML when possible so layout styles stick.
    """
    paras = list(tf.paragraphs)
    # Ensure we have enough paragraphs by cloning the last one
    while len(paras) < len(items):
        last = paras[-1]._p
        new_p = copy.deepcopy(last)
        last.addnext(new_p)
        paras = list(tf.paragraphs)

    # Remove excess paragraphs
    for p in paras[len(items) :]:
        p._p.getparent().remove(p._p)
    paras = list(tf.paragraphs)

    for p, (level, text) in zip(paras, items):
        p.level = level
        if p.runs:
            # keep first run formatting
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
            p.runs[0].text = text
        else:
            p.text = text


def set_ph(slide, idx: int, text: str | None = None, bullets: list[tuple[int, str]] | None = None) -> None:
    ph = slide.placeholders[idx]
    tf = ph.text_frame
    if bullets is not None:
        clear_and_bullets(tf, bullets)
    elif text is not None:
        lines = text.split("\n")
        clear_and_bullets(tf, [(0, line) for line in lines])


def add_slide(prs: Presentation, layout_name: str):
    layout = next(l for l in prs.slide_layouts if l.name == layout_name)
    return prs.slides.add_slide(layout)


def build() -> None:
    shutil.copy2(TEMPLATE, OUTPUT)
    prs = Presentation(str(OUTPUT))
    delete_all_slides(prs)

    # ---- 1 Title ----
    s = add_slide(prs, "SQ_TITLE")
    set_ph(s, 110, "CS490  ·  DiSC Lab  ·  UNLV")
    set_ph(s, 0, TITLE)
    set_ph(s, 111, "Graph construction choices for binary review helpfulness prediction")
    set_ph(s, 112, "Similarity graphs, bipartite structure, and controlled GNN experiments")
    set_ph(s, 113, "Katya Alimov  ·  CS490  ·  alimok1@unlv.nevada.edu")
    set_ph(
        s,
        114,
        "Department of Computer Science  ·  University of Nevada, Las Vegas\n"
        "Data-intensive Scalable Computing (DiSC) Lab",
    )

    # ---- 2 Roadmap ----
    s = add_slide(prs, "SQ_NUMBERED")
    set_ph(s, 0, "Roadmap")
    set_ph(s, 101, "What this talk covers — problem through insights")
    set_ph(s, 110, bullets=[(0, "Problem"), (1, "Why review helpfulness matters for renters and platforms")])
    set_ph(s, 111, bullets=[(0, "Background"), (1, "Data, labels, and related graph / GNN work")])
    set_ph(s, 112, bullets=[(0, "Method"), (1, "Two graphs, four models, fair seeded splits")])
    set_ph(s, 113, bullets=[(0, "Experiments"), (1, "Threshold density, rating filter, weighted edges next")])
    set_ph(s, 114, bullets=[(0, "Results & insights"), (1, "What changed performance — and what did not")])

    # ---- 3 Motivation ----
    s = add_slide(prs, "SQ_CONTENT")
    set_ph(s, 0, "The problem")
    set_ph(s, 101, "Apartment reviews vary widely in usefulness; platforms need ranking signals")
    set_ph(
        s,
        110,
        bullets=[
            (0, "Task: predict whether a review is helpful or not helpful (binary)"),
            (0, "Helpful reviews reduce search cost; noisy reviews hide signal"),
            (0, "Text alone misses structure — which reviews and apartments relate"),
            (0, "Prior rental helpfulness work (Hassan et al., 2026) stops at tabular / transformer baselines"),
            (0, "Claim: modeling reviews as a graph can expose relational signal beyond isolated text"),
        ],
    )

    # ---- 4 Background & data ----
    s = add_slide(prs, "SQ_DEFINITION")
    set_ph(s, 0, "Background and data")
    set_ph(s, 101, "Dataset plus the literature that shapes the design")
    set_ph(
        s,
        110,
        bullets=[
            (0, "Dataset"),
            (1, "~40K apartment reviews with text, ratings, property metadata"),
            (1, "Binary helpfulness labels via temporal-adjusted k-means (Hassan et al.)"),
            (1, "Features: Sentence-BERT embeddings + handcrafted text / listing signals"),
        ],
    )
    set_ph(
        s,
        111,
        bullets=[
            (0, "Related work"),
            (1, "GCN (Kipf & Welling) — spectral message passing baseline"),
            (1, "GraphSAGE (Hamilton) — inductive aggregators for new reviews"),
            (1, "Chen (2021) — heterogeneous user–review–product graphs beat text-only"),
            (1, "Yao / TextGCN — document graphs as a blueprint for review similarity"),
        ],
    )
    set_ph(
        s,
        112,
        bullets=[
            (0, "Takeaway for this project"),
            (1, "Relational modeling helps, but text-similarity edges are costly and sensitive to how we connect nodes — that is the design space we test."),
        ],
    )

    # ---- 5 Technical challenges ----
    s = add_slide(prs, "SQ_CARDS3")
    set_ph(s, 0, "Technical challenges")
    set_ph(s, 101, "Three bottlenecks that dominate graph construction and training")
    set_ph(
        s,
        110,
        bullets=[
            (0, "Edge decisions"),
            (1, "Cosine threshold changes density by orders of magnitude"),
            (1, "Too dense → noise / cost; too sparse → isolated nodes"),
        ],
    )
    set_ph(
        s,
        111,
        bullets=[
            (0, "Fair comparison"),
            (1, "Same seed, same train/val/test masks across graphs"),
            (1, "Only the edges (or model) should change"),
        ],
    )
    set_ph(
        s,
        112,
        bullets=[
            (0, "Scale"),
            (1, "t=0.60 yields ~40M edges — slow and memory-heavy"),
            (1, "Similarity graphs need chunked computation"),
        ],
    )

    # ---- 6 Approach overview ----
    s = add_slide(prs, "SQ_TWOPANEL")
    set_ph(s, 0, "Approach overview")
    set_ph(s, 101, "Two graphs, shared labels, GNN (and MLP) classifiers")
    set_ph(
        s,
        110,
        bullets=[
            (0, "Graph 1 — review similarity"),
            (1, "Nodes = reviews"),
            (1, "Edge if text cosine ≥ threshold t"),
            (1, "Homogeneous GCN / SAGE / GAT / MLP"),
        ],
    )
    set_ph(
        s,
        111,
        bullets=[
            (0, "Graph 2 — bipartite"),
            (1, "Review ↔ apartment relations"),
            (1, "Heterogeneous message passing (HeteroSAGE)"),
            (1, "Captures listing context, not only peer reviews"),
        ],
    )

    # ---- 7 Graph 1 ----
    s = add_slide(prs, "SQ_CONTENT")
    set_ph(s, 0, "Graph 1 — review–review similarity")
    set_ph(s, 101, "Each node is one review; edges encode “similar enough to connect”")
    set_ph(
        s,
        110,
        bullets=[
            (0, "Node features x: SBERT + handcrafted signals"),
            (0, "Label y: helpful (1) / not helpful (0)"),
            (0, "edge_index: undirected links where cosine(SBERT_i, SBERT_j) ≥ t"),
            (0, "Lower t → denser graph; higher t → sparser graph"),
            (0, "Hypothesis: mid thresholds beat very low and very high ones"),
            (0, "Observed edges: t=0.60 → 40.2M; 0.75 → 684K; 0.80 → 93K; 0.90 → 3.9K"),
        ],
    )

    # ---- 8 Graph 2 ----
    s = add_slide(prs, "SQ_CONTENT")
    set_ph(s, 0, "Graph 2 — bipartite review–apartment")
    set_ph(s, 101, "Reviews connect to the apartments they describe — structure without cosine thresholds")
    set_ph(
        s,
        110,
        bullets=[
            (0, "Node types: review and apartment, each with its own feature matrix"),
            (0, "Edges: review → belongs_to → apartment (and reverse)"),
            (0, "Prediction stays on review nodes (helpfulness labels)"),
            (0, "Motivation from literature: interaction graphs outperform isolated text (Chen, 2021)"),
            (0, "GCN’s undirected design needs a heterogeneous / bipartite workaround for this mapping"),
        ],
    )

    # ---- 9 Models ----
    s = add_slide(prs, "SQ_CARDS3")
    set_ph(s, 0, "Models")
    set_ph(s, 101, "Baselines share the same split; Graph 1 vs Graph 2 architectures differ")
    set_ph(
        s,
        110,
        bullets=[
            (0, "No-graph control"),
            (1, "MLP on review features only"),
            (1, "Asks: do edges help at all?"),
        ],
    )
    set_ph(
        s,
        111,
        bullets=[
            (0, "Graph 1 GNNs"),
            (1, "GCN — spectral neighborhood mix"),
            (1, "GraphSAGE — sampled aggregation"),
            (1, "GAT — learned attention over neighbors"),
        ],
    )
    set_ph(
        s,
        112,
        bullets=[
            (0, "Graph 2 GNN"),
            (1, "HeteroSAGE on review↔apartment"),
            (1, "Type-aware message passing"),
        ],
    )

    # ---- 10 Baseline results ----
    s = add_slide(prs, "SQ_TABLE")
    set_ph(s, 0, "Baseline results")
    set_ph(s, 101, "Fixed seed-42 split; Macro F1 is the primary score")
    set_ph(
        s,
        110,
        bullets=[
            (0, "Model                          Graph     Acc      Macro F1"),
            (0, "MLP (no graph)                 Graph 1   0.700    0.700"),
            (0, "GCN                             Graph 1   0.693    0.693"),
            (0, "GraphSAGE                       Graph 1   0.689    0.689"),
            (0, "GAT                             Graph 1   0.690    0.690"),
            (0, "HeteroSAGE                      Graph 2   0.709    0.709"),
        ],
    )
    set_ph(
        s,
        111,
        "Graph 2 HeteroSAGE leads; on default Graph 1 edges, MLP slightly beats homogeneous GNNs — edge quality matters.",
    )

    # ---- 11 Exp 1 setup ----
    s = add_slide(prs, "SQ_CONTENT")
    set_ph(s, 0, "Experiment 1 — cosine threshold")
    set_ph(s, 101, "Change only Graph 1 edges; keep masks, features, and GCN fixed")
    set_ph(
        s,
        110,
        bullets=[
            (0, "Sweep t ∈ {0.75, 0.80, 0.90} with GCN (skip 0.60 — too dense / too slow)"),
            (0, "Same 70/15/15 stratified split (seed 42) on every run"),
            (0, "Report Accuracy, Macro F1, and best validation accuracy"),
            (0, "Question: does denser always help message passing?"),
            (0, "Density span: 683,692 edges (t=0.75) down to 3,942 (t=0.90)"),
        ],
    )

    # ---- 12 Exp 1 results ----
    s = add_slide(prs, "SQ_TABLE")
    set_ph(s, 0, "Experiment 1 results")
    set_ph(s, 101, "GCN on Graph 1 — only the threshold changes")
    set_ph(
        s,
        110,
        bullets=[
            (0, "Threshold t     Edges        Acc      Macro F1   Best Val Acc"),
            (0, "0.75            683,692      0.668    0.668      0.667"),
            (0, "0.80             92,866      0.676    0.676      0.676"),
            (0, "0.90              3,942      0.651    0.651      0.650"),
        ],
    )
    set_ph(
        s,
        111,
        "Best here: t=0.80. More edges ≠ better — t=0.90 is too sparse; gaps vs 0.75 are mild.",
    )

    # ---- 13 Exp 2 setup ----
    s = add_slide(prs, "SQ_CONTENT")
    set_ph(s, 0, "Experiment 2 — text AND rating filter")
    set_ph(s, 101, "Connect reviews only when text is similar and ratings agree enough")
    set_ph(
        s,
        110,
        bullets=[
            (0, "AND rule: cosine ≥ t  AND  |Δrating| ≤ d"),
            (0, "Test d=2 (mild) and d=1 (strict) against text-only edges"),
            (0, "Same GCN setup and masks as Experiment 1"),
            (0, "Motivation: similar text with very different stars may be weak neighbors"),
            (0, "Edge counts drop only slightly under AND — filter is selective, not drastic"),
        ],
    )

    # ---- 14 Exp 2 results ----
    s = add_slide(prs, "SQ_TABLE")
    set_ph(s, 0, "Experiment 2 results")
    set_ph(s, 101, "Macro F1 — text-only vs AND rating filters (GCN)")
    set_ph(
        s,
        110,
        bullets=[
            (0, "t        Text-only    AND d=2     AND d=1"),
            (0, "0.75     0.668        0.684       0.681"),
            (0, "0.80     0.676        0.686       0.663"),
            (0, "0.90     0.651        0.669       0.669"),
            (0, ""),
            (0, "Best overall in this set: AND d=2 at t=0.80 (F1 0.686)"),
        ],
    )
    set_ph(
        s,
        111,
        "Mild rating agreement (d=2) beats text-only at every t; stricter d=1 can hurt (esp. t=0.80).",
    )

    # ---- 15 Exp 3 placeholder ----
    s = add_slide(prs, "SQ_CONTENT")
    set_ph(s, 0, "Experiment 3 — weighted edges (next)")
    set_ph(s, 101, "Replace binary edges with continuous similarity weights")
    set_ph(
        s,
        110,
        bullets=[
            (0, "Status: planned — results will replace this placeholder"),
            (0, "Idea: keep an edge when cosine ≥ t, but weight it by the similarity score"),
            (0, "Compare against binary Graph 1 and best AND setting"),
            (0, "Question: does soft weighting beat hard on/off neighbors?"),
            (0, "Same masks and GCN protocol for a fair head-to-head"),
        ],
    )

    # ---- 16 Insights ----
    s = add_slide(prs, "SQ_STATS")
    set_ph(s, 0, "Insights and next steps")
    set_ph(s, 101, "What the experiments say so far")
    set_ph(
        s,
        110,
        bullets=[
            (0, "Graph structure is a design choice — not a free win over MLP"),
            (0, "Bipartite HeteroSAGE (0.709 F1) currently leads overall baselines"),
            (0, "On similarity graphs, mid threshold + mild rating filter works best"),
            (0, "Next: finish weighted edges, then revisit whether Graph 1 can close the Graph 2 gap"),
        ],
    )
    set_ph(s, 111, bullets=[(0, "0.709"), (1, "Best F1 — HeteroSAGE")])
    set_ph(s, 112, bullets=[(0, "0.686"), (1, "Best AND GCN F1")])
    set_ph(s, 113, bullets=[(0, "0.80"), (1, "Best cosine threshold")])
    set_ph(s, 114, bullets=[(0, "d=2"), (1, "Best rating window")])

    # ---- 17 Thank you ----
    s = add_slide(prs, "SQ_CLOSING")
    set_ph(s, 0, "Thank you")
    set_ph(s, 110, "Questions welcome")
    set_ph(s, 111, "alimok1@unlv.nevada.edu\nDiSC Lab · UNLV")
    set_ph(s, 112, "CS490  ·  Advising: DiSC Lab")

    prs.save(str(OUTPUT))
    print(f"Wrote {OUTPUT} with {len(prs.slides)} slides")


if __name__ == "__main__":
    build()
