from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


FALLBACK_DOCS = [
    "Warfarin and Aspirin together may increase bleeding risk.",
    "Ibuprofen may reduce the effectiveness of Lisinopril and increase kidney complications.",
    "Metformin combined with Alcohol may increase lactic acidosis risk.",
    "Sildenafil and Nitroglycerin together can dangerously lower blood pressure.",
    "Paracetamol and Alcohol together may increase liver damage risk.",
    "Aspirin and Clopidogrel together may cause excessive bleeding.",
]


def load_pubmed_documents(max_docs=975, docs_dir="medical_docs"):
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        return FALLBACK_DOCS, "fallback"

    documents = []

    for file_path in sorted(docs_path.glob("*.txt"))[:max_docs]:
        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if text:
            documents.append(
                f"{file_path.name}\n{text[:3500]}"
            )

    return (
        documents or FALLBACK_DOCS,
        "pubmed" if documents else "fallback"
    )


def build_vector_index(documents):
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    doc_embeddings = embedding_model.encode(
        documents,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    doc_embeddings = np.asarray(
        doc_embeddings,
        dtype="float32"
    )

    return embedding_model, doc_embeddings


def search_documents(
    query,
    documents,
    model,
    index,
    top_k=5
):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        index
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    return [
        documents[idx][:900]
        for idx in top_indices
    ]