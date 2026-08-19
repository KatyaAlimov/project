"""Export a 1–2 page Methods draft PDF."""

from fpdf import FPDF


BODY = [
    (
        "Research design",
        "This study uses a computational, observational design. We do not assign treatments or collect new human ratings. The unit of analysis is the review. The outcome is a binary helpfulness label already stored in the corpus (kmeans_label). We encode the same reviews as (a) feature vectors only, (b) a homogeneous review-review similarity graph (Graph 1), and (c) a heterogeneous review-apartment bipartite graph (Graph 2), then compare models under one shared train/validation/test split. The design tests whether graph structure adds predictive information beyond node features.",
    ),
    (
        "Research questions, hypotheses, and objectives",
        "Objectives. (1) Represent each review with linguistic, rating, and market-context features plus a sentence embedding. (2) Construct Graph 1 (review similarity) and Graph 2 (review-apartment). (3) Train baseline classifiers under a shared protocol and report accuracy, macro-F1, precision, and recall. (4) Test whether edges improve prediction over a feature-only model.\n\n"
        "Research questions. RQ1: How well can binary helpfulness be predicted from node features without graph edges (MLP)? RQ2: Does message passing on the review-similarity graph (GCN, GraphSAGE, GAT) improve over the MLP? RQ3: Does a heterogeneous GNN on the review-apartment graph (HeteroSAGE) improve over the MLP and over Graph 1?\n\n"
        "Hypotheses. H1: Review-level text and metadata features predict the binary label above chance. H2: Similarity edges help only if they carry signal not already in the SBERT embedding; otherwise Graph 1 GNNs will not beat the MLP. H3: Apartment-level neighborhoods help if helpfulness is partly a listing/community property rather than isolated review text.\n\n"
        "[Draft note: Match RQ/H wording to the Introduction. If the Introduction does not state H2/H3, keep objectives + RQs only.]",
    ),
    (
        "Terms and variables",
        "Review helpfulness (dependent variable): binary kmeans_label in {0, 1}. [Draft note: state how k-means was fit and how clusters map to helpful vs not; this is not implemented in the graph-building script.]\n"
        "Review node: one row after dropping missing ZIP (required for market tier); N is approximately 40,671. Apartment node: unique listing identified by URL.\n"
        "Graph 1: undirected review-review graph; an edge if SBERT cosine similarity is at least 0.75. Graph 2: bipartite review and apartment nodes; each review has a belongs-to / has-review link to its listing.\n"
        "Feature-only baseline: MLP on the same review vector with no edges (structure ablation).\n\n"
        "Review predictors (about 391 dimensions): review length (character count); rating deviation (|rating - listing average|); TextBlob polarity and subjectivity; lexical diversity (type-token ratio); imperative density (share of sentences whose first POS tag is VB); ZIP-level market tier from price per square foot (0 = budget, 1 = standard, 2 = luxury); 384-dimensional all-MiniLM-L6-v2 embedding. Apartment features (Graph 2): average rating, midpoint rent, midpoint square footage, market tier.\n\n"
        "Design controls (not causal): stratified 70/15/15 split; train-only z-scoring; seed 42; early stopping on validation accuracy.",
    ),
    (
        "Limitations to generalizability",
        "(1) Domain: U.S. apartment reviews from Apartments.com-style listings; not assumed to transfer to other platforms or products. (2) Outcome: k-means-derived proxy, not verified helpful clicks or new human labels. (3) Selection: missing-ZIP reviews dropped; reviewers are not a random sample of renters. (4) Graph 1 threshold (0.75) is a design choice; similarity edges may be redundant with SBERT. (5) One split and fixed hyperparameters (hidden size 64, two layers, dropout 0.3, Adam lr = 0.01, weight decay 5e-4, max 100 epochs, patience 20). (6) Transductive full-batch training: Graph 1 edges use embeddings of all reviews, so validation/test neighbors can influence training. (7) English-oriented TextBlob/NLTK features. (8) Results can vary slightly by device and library version.",
    ),
    (
        "Minimizing these limitations",
        "All models use the same features, labels, and split. The MLP is the structure ablation for RQ2/RQ3. The split is stratified; feature means and standard deviations come from training nodes only. Early stopping uses validation accuracy; test metrics are from the best validation epoch. We report macro-F1 as well as accuracy. Graph 2 edges use positional node IDs after resetting the table index so links align with feature rows.\n\n"
        "[Draft note if time before the paper deadline: inductive/masked Graph 1 edges; threshold sensitivity; repeated splits; a class-balance table; a small human audit of kmeans_label.]",
    ),
    (
        "Data collection and analysis",
        "Collection. This draft analyzes an existing table (kmeans_binary_labeled_helpfulness_data_final.csv). Empty review text is filled with an empty string; rows without ZIP are excluded. [Draft note: add original scrape or dump dates, source, and any ethics/IRB or terms-of-use statement required by the course.]\n\n"
        "Construction. Rent strings are parsed to numeric values; market tier is computed within ZIP. Linguistic features and SBERT embeddings are computed from review text. Graph 1 drops self-loops. Graph 2 stores both edge directions for message passing.\n\n"
        "Analysis. PyTorch Geometric, full-batch node classification, cross-entropy, Adam as above. Models: MLP; GCN; GraphSAGE (mean); GAT (4 heads then 1); HeteroSAGE (two HeteroConv layers). Test metrics: accuracy, macro-F1, macro precision, macro recall. Numeric results belong in Results, not Methods.",
    ),
    (
        "Summary of research methods and design",
        "This is a comparative graph-machine-learning study of binary review-helpfulness prediction on one observational corpus. The same reviews are represented as feature vectors, a text-similarity graph, and a review-apartment bipartite graph. MLP, homogeneous GNNs, and HeteroSAGE are trained under one stratified split, train-only scaling, and early stopping. The design asks whether structure helps once content and listing context are already in the node features, with limits of platform, proxy labels, and a single transductive split.",
    ),
]


class MethodsPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            self.set_font("Times", "B", 14)
            self.cell(0, 8, "METHODS", new_x="LMARGIN", new_y="NEXT", align="C")
            self.set_font("Times", "I", 9)
            self.cell(
                0,
                5,
                "Draft  |  1-2 pages  |  Align wording with the Introduction before submission",
                new_x="LMARGIN",
                new_y="NEXT",
                align="C",
            )
            self.ln(2)
            self.set_draw_color(40, 40, 40)
            self.line(18, self.get_y(), 198, self.get_y())
            self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Times", "I", 8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, f"{self.page_no()}", align="C")


def main():
    pdf = MethodsPDF(format="Letter", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(18, 14, 18)
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)

    for title, text in BODY:
        pdf.set_font("Times", "B", 11)
        pdf.multi_cell(0, 5.2, title)
        pdf.ln(0.6)
        pdf.set_font("Times", "", 10)
        pdf.multi_cell(0, 4.5, text)
        pdf.ln(2.2)

    out = "/Users/alimok1/Documents/project/Methods_Draft.pdf"
    pdf.output(out)
    print(f"pages={pdf.page_no()}")
    print(out)


if __name__ == "__main__":
    main()
