from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path


def load_documents(max_docs=975):
    docs_path = Path("medical_docs")

    fallback_docs = [
        "Warfarin and Aspirin together may increase bleeding risk.",
        "Ibuprofen may reduce the effectiveness of Lisinopril and increase kidney complications.",
        "Metformin combined with Alcohol may increase lactic acidosis risk.",
        "Sildenafil and Nitroglycerin together can dangerously lower blood pressure.",
        "Paracetamol and Alcohol together may increase liver damage risk.",
        "Aspirin and Clopidogrel together may cause excessive bleeding.",
    ]

    if not docs_path.exists():
        return fallback_docs

    docs = []

    for file_path in sorted(docs_path.glob("*.txt"))[:max_docs]:
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()

        if text:
            docs.append(f"{file_path.name}\n{text[:3500]}")

    return docs or fallback_docs


docs = load_documents()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(docs)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings, dtype="float32"))

print(f"Vector store created successfully with {len(docs)} documents.")
