"""
Synthetic academic research PDF generator for ResearchLens AI testing and verification.
Generates 3 multi-page papers with known sections, metadata, results tables, limitations,
and future work items to ensure reproducible testing.
"""

from pathlib import Path

import pymupdf


def create_paper_1(output_path: str) -> None:
    """
    Paper 1: Attention-Based Deep Learning for Medical Document Intelligence
    Topics: NLP, Transformers, Healthcare, Classification
    Dataset: PubMedMedDoc-50k
    Metric: Accuracy (94.2%), Macro-F1 (93.8%)
    Limitations: Restricted to English medical summaries; sample size 5,000.
    Future Work: Multilingual clinical notes, self-supervised pre-training.
    """
    doc = pymupdf.open()

    # Page 1: Title, Authors, Abstract, Introduction
    page1 = doc.new_page(width=595, height=842)  # A4
    page1.insert_text(
        (50, 60),
        "Attention-Based Deep Learning for Medical Document Intelligence",
        fontsize=16,
        fontname="helv",
    )
    page1.insert_text(
        (50, 85),
        "Dr. Eleanor Vance, Dr. Marcus Brody, Dr. Henry Wu\nDepartment of Computer Science, Stanford University\nDOI: 10.1145/3456789.0123456",
        fontsize=10,
    )

    page1.insert_text((50, 140), "Abstract", fontsize=12, fontname="helv")
    abstract_text = (
        "Automated understanding of clinical and medical documents requires high-precision semantic parsing. "
        "In this paper, we introduce MedLens-Transformer, a specialized multi-head self-attention architecture "
        "tailored for clinical documentation analysis. Our model processes long-range dependencies across complex EHR "
        "records and unstructured clinical narratives. Evaluated on the newly released PubMedMedDoc-50k benchmark, "
        "MedLens-Transformer establishes a new state-of-the-art with 94.2% classification accuracy and 93.8% Macro-F1 score."
    )
    page1.insert_textbox(pymupdf.Rect(50, 155, 545, 240), abstract_text, fontsize=9.5)

    page1.insert_text((50, 260), "1. Introduction", fontsize=12, fontname="helv")
    intro_text = (
        "Medical records represent one of the richest yet most challenging frontiers in natural language processing. "
        "Clinical narratives contain non-standard abbreviations, domain-specific terminology, and intricate syntactic structures. "
        "Traditional recurrent networks struggle with vanishing gradients when processing extended clinical encounters. "
        "Our objective in this research is to design an attention-only neural architecture capable of capturing subtle diagnostic clues "
        "without requiring prohibitive recurrent computations."
    )
    page1.insert_textbox(pymupdf.Rect(50, 275, 545, 380), intro_text, fontsize=9.5)

    page1.insert_text((50, 400), "2. Related Work", fontsize=12, fontname="helv")
    rel_text = (
        "Prior work in medical document intelligence predominantly relied on convolutional neural networks and BiLSTM architectures. "
        "Devlin et al. (2019) demonstrated the transformative impact of bidirectional transformers on general language understanding. "
        "BioBERT and ClinicalBERT adapted these paradigms to biomedical texts through masked language modeling."
    )
    page1.insert_textbox(pymupdf.Rect(50, 415, 545, 500), rel_text, fontsize=9.5)

    # Page 2: Methodology, Algorithms, Datasets
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 60), "3. Methodology and Architecture", fontsize=12, fontname="helv")
    method_text = (
        "The MedLens-Transformer consists of a 12-layer bidirectional encoder with 768 hidden dimensions and 12 attention heads. "
        "We employ scaled dot-product attention with specialized relative positional encodings designed for hierarchical document structures. "
        "Token embeddings are augmented with biomedical entity type indicators extracted using UMLS concept mapping."
    )
    page2.insert_textbox(pymupdf.Rect(50, 75, 545, 175), method_text, fontsize=9.5)

    page2.insert_text((50, 195), "3.1 Algorithms and Models", fontsize=11, fontname="helv")
    algo_text = (
        "We deploy the AdamW optimizer with a learning rate of 2e-5, linear warmup over the first 10% of training steps, "
        "and cosine learning rate decay. Cross-entropy loss with label smoothing (epsilon = 0.1) is utilized to prevent overconfidence."
    )
    page2.insert_textbox(pymupdf.Rect(50, 210, 545, 290), algo_text, fontsize=9.5)

    page2.insert_text((50, 310), "4. Datasets and Experimental Setup", fontsize=12, fontname="helv")
    dataset_text = (
        "Our experimental evaluation leverages the PubMedMedDoc-50k benchmark, comprising 50,000 annotated clinical discharge summaries "
        "categorized into 25 diagnostic specialties. Data was partitioned into 80% training, 10% validation, and 10% test splits."
    )
    page2.insert_textbox(pymupdf.Rect(50, 325, 545, 410), dataset_text, fontsize=9.5)

    # Page 3: Results, Discussion, Limitations, Future Work
    page3 = doc.new_page(width=595, height=842)
    page3.insert_text((50, 60), "5. Experimental Results", fontsize=12, fontname="helv")
    res_text = (
        "Quantitative results on the test partition of PubMedMedDoc-50k show that MedLens-Transformer outperforms all baselines. "
        "Specifically, MedLens-Transformer achieved an accuracy of 94.2% and a Macro-F1 score of 93.8%, exceeding ClinicalBERT by 3.4 percentage points."
    )
    page3.insert_textbox(pymupdf.Rect(50, 75, 545, 150), res_text, fontsize=9.5)

    page3.insert_text((50, 170), "6. Limitations", fontsize=12, fontname="helv")
    lim_text = (
        "Our evaluation is restricted to English medical summaries and was tested only on 5,000 samples due to computational resource constraints. "
        "Furthermore, our tokenization vocabulary does not fully cover handwritten clinician annotations."
    )
    page3.insert_textbox(pymupdf.Rect(50, 185, 545, 260), lim_text, fontsize=9.5)

    page3.insert_text((50, 280), "7. Future Work", fontsize=12, fontname="helv")
    fut_text = "Future work will extend the architecture to multilingual clinical notes and evaluate self-supervised pre-training across multi-modal records."
    page3.insert_textbox(pymupdf.Rect(50, 295, 545, 360), fut_text, fontsize=9.5)

    page3.insert_text((50, 380), "8. Conclusion", fontsize=12, fontname="helv")
    conc_text = "We presented MedLens-Transformer, demonstrating that relative attention mechanisms provide substantial advantages for clinical NLP tasks."
    page3.insert_textbox(pymupdf.Rect(50, 395, 545, 460), conc_text, fontsize=9.5)

    page3.insert_text((50, 480), "References", fontsize=12, fontname="helv")
    ref_text = (
        "[1] Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.\n"
        "[2] Alsentzer, E., et al. (2019). Publicly Available Clinical BERT Embeddings. BioNLP Workshop.\n"
        "[3] Lee, J., et al. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics."
    )
    page3.insert_textbox(pymupdf.Rect(50, 495, 545, 600), ref_text, fontsize=8.5)

    doc.save(output_path)
    doc.close()


def create_paper_2(output_path: str) -> None:
    """
    Paper 2: Graph Neural Networks for Scientific Citation Link Prediction
    Topics: Graph Neural Networks, Citation Analysis, Link Prediction
    Dataset: Cora-Citation-Graph
    Metric: MRR (0.784), Hit@10 (86.5%)
    Limitations: High GPU memory overhead; scalability to millions of nodes remains an open issue.
    Future Work: Subgraph sampling algorithms and temporal dynamic link prediction.
    """
    doc = pymupdf.open()

    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 60),
        "Graph Neural Networks for Scientific Citation Link Prediction",
        fontsize=16,
        fontname="helv",
    )
    page1.insert_text(
        (50, 85),
        "Dr. Julian Thorne, Dr. Sarah Lin\nOxford Institute for AI, University of Oxford\nDOI: 10.1109/ICDM.2024.9876543",
        fontsize=10,
    )

    page1.insert_text((50, 140), "Abstract", fontsize=12, fontname="helv")
    abstract_text = (
        "Predicting missing citation links in scientific bibliographies is vital for tracking literature trends. "
        "In this study, we propose CiteGraph-GNN, an inductive relational graph attention network capable of capturing "
        "multi-hop topological neighborhoods alongside document title semantics. "
        "Tested on the standard Cora-Citation-Graph benchmark, CiteGraph-GNN achieves an MRR of 0.784 and Hit@10 of 86.5%."
    )
    page1.insert_textbox(pymupdf.Rect(50, 155, 545, 240), abstract_text, fontsize=9.5)

    page1.insert_text((50, 260), "1. Introduction", fontsize=12, fontname="helv")
    intro_text = (
        "Academic literature expands exponentially, making manual tracking of citation relationships untenable. "
        "Our objective is to accurately forecast prospective citation ties between papers using structural and content features. "
        "We model the academic library as a heterogeneous citation graph."
    )
    page1.insert_textbox(pymupdf.Rect(50, 275, 545, 360), intro_text, fontsize=9.5)

    page1.insert_text((50, 380), "2. Methodology", fontsize=12, fontname="helv")
    method_text = (
        "CiteGraph-GNN computes edge attention coefficients between papers using transformed feature vectors. "
        "Each node aggregates representations from k-hop neighborhoods with multi-head attention weights."
    )
    page1.insert_textbox(pymupdf.Rect(50, 395, 545, 480), method_text, fontsize=9.5)

    # Page 2: Datasets, Results, Limitations, Future Work
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 60), "3. Datasets and Evaluation", fontsize=12, fontname="helv")
    exp_text = (
        "We test CiteGraph-GNN on the Cora-Citation-Graph benchmark, consisting of 2,708 scientific publications and 5,429 citation links. "
        "Evaluation is measured via Mean Reciprocal Rank (MRR) and Hit@10."
    )
    page2.insert_textbox(pymupdf.Rect(50, 75, 545, 150), exp_text, fontsize=9.5)

    page2.insert_text((50, 170), "4. Results", fontsize=12, fontname="helv")
    res_text = "CiteGraph-GNN achieves an MRR score of 0.784 and Hit@10 of 86.5% on Cora-Citation-Graph, outperforming standard GCN by 5.2%."
    page2.insert_textbox(pymupdf.Rect(50, 185, 545, 250), res_text, fontsize=9.5)

    page2.insert_text((50, 270), "5. Limitations", fontsize=12, fontname="helv")
    lim_text = "The model requires high GPU memory for dense subgraphs and scalability to millions of nodes remains an open issue."
    page2.insert_textbox(pymupdf.Rect(50, 285, 545, 340), lim_text, fontsize=9.5)

    page2.insert_text((50, 360), "6. Future Work", fontsize=12, fontname="helv")
    fut_text = "Future work will formulate subgraph sampling algorithms to enable training on large-scale billion-edge knowledge graphs."
    page2.insert_textbox(pymupdf.Rect(50, 375, 545, 430), fut_text, fontsize=9.5)

    page2.insert_text((50, 450), "References", fontsize=12, fontname="helv")
    ref_text = (
        "[1] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.\n"
        "[2] Velickovic, P., et al. (2018). Graph Attention Networks. ICLR."
    )
    page2.insert_textbox(pymupdf.Rect(50, 465, 545, 550), ref_text, fontsize=8.5)

    doc.save(output_path)
    doc.close()


def create_paper_3(output_path: str) -> None:
    """
    Paper 3: Hybrid Information Retrieval for Scientific Document Search
    Topics: Information Retrieval, BM25, Dense Embeddings, RRF
    Dataset: MS-MARCO-Academic
    Metric: nDCG@10 (0.642), MRR@10 (0.681)
    Limitations: Reranking step introduces query latency unsuitable for real-time mobile apps.
    Future Work: Quantization and speculative early-exit cascades for low-latency reranking.
    """
    doc = pymupdf.open()

    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 60),
        "Hybrid Information Retrieval for Scientific Document Search",
        fontsize=16,
        fontname="helv",
    )
    page1.insert_text(
        (50, 85),
        "Dr. Aris Thorne, Dr. Maya Patel\nMIT CSAIL\nDOI: 10.1145/9988776.1122334",
        fontsize=10,
    )

    page1.insert_text((50, 140), "Abstract", fontsize=12, fontname="helv")
    abstract_text = (
        "Modern academic search requires balancing exact lexical matching of technical terms with semantic generalization. "
        "We introduce HybridLens, combining Okapi BM25 and dense bi-encoder embeddings via Reciprocal Rank Fusion (RRF) "
        "coupled with a Cross-Encoder reranker. Evaluated on MS-MARCO-Academic, HybridLens achieves nDCG@10 of 0.642 and MRR@10 of 0.681."
    )
    page1.insert_textbox(pymupdf.Rect(50, 155, 545, 240), abstract_text, fontsize=9.5)

    page1.insert_text((50, 260), "1. Introduction", fontsize=12, fontname="helv")
    intro_text = (
        "Dense retrieval models often suffer from vocabulary mismatch on specialized jargon, while sparse lexical models fail on synonymy. "
        "Our objective is to demonstrate that reciprocal rank fusion yields superior retrieval robustness across heterogeneous scientific disciplines."
    )
    page1.insert_textbox(pymupdf.Rect(50, 275, 545, 360), intro_text, fontsize=9.5)

    # Page 2: Methods, Results, Limitations, Future Work
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 60), "2. Methodology and Algorithms", fontsize=12, fontname="helv")
    method_text = (
        "HybridLens fuses sparse scores from BM25 with dense inner product scores from BAAI/bge-small-en-v1.5 using RRF with constant k=60. "
        "Top-20 candidates are then reranked by an ms-marco-MiniLM cross-encoder."
    )
    page2.insert_textbox(pymupdf.Rect(50, 75, 545, 160), method_text, fontsize=9.5)

    page2.insert_text((50, 180), "3. Results", fontsize=12, fontname="helv")
    res_text = "On MS-MARCO-Academic, HybridLens achieved an nDCG@10 of 0.642 and MRR@10 of 0.681, surpassing dense-only retrieval by 4.8%."
    page2.insert_textbox(pymupdf.Rect(50, 195, 545, 270), res_text, fontsize=9.5)

    page2.insert_text((50, 290), "4. Limitations", fontsize=12, fontname="helv")
    lim_text = "The cross-encoder reranking step introduces query latency unsuitable for real-time mobile applications."
    page2.insert_textbox(pymupdf.Rect(50, 305, 545, 360), lim_text, fontsize=9.5)

    page2.insert_text((50, 380), "5. Future Work", fontsize=12, fontname="helv")
    fut_text = "Future work will investigate vector quantization and speculative early-exit cascades to reduce reranker computational complexity."
    page2.insert_textbox(pymupdf.Rect(50, 395, 545, 450), fut_text, fontsize=9.5)

    page2.insert_text((50, 470), "References", fontsize=12, fontname="helv")
    ref_text = (
        "[1] Cormack, G. V., et al. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR.\n"
        "[2] Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Information Retrieval."
    )
    page2.insert_textbox(pymupdf.Rect(50, 485, 545, 560), ref_text, fontsize=8.5)

    doc.save(output_path)
    doc.close()


def generate_all_sample_pdfs(fixtures_dir: str = "tests/fixtures") -> None:
    """Generates all 3 sample research paper PDFs in the fixtures directory."""
    path = Path(fixtures_dir)
    path.mkdir(parents=True, exist_ok=True)
    p1 = str(path / "paper_1_medical_nlp.pdf")
    p2 = str(path / "paper_2_citation_gnn.pdf")
    p3 = str(path / "paper_3_hybrid_ir.pdf")

    create_paper_1(p1)
    create_paper_2(p2)
    create_paper_3(p3)
    print(f"Sample research PDFs generated:\n - {p1}\n - {p2}\n - {p3}")


if __name__ == "__main__":
    generate_all_sample_pdfs()
