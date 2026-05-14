from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Documents
docs = [
    "Warfarin and Aspirin together may increase bleeding risk.",
    "Ibuprofen may reduce the effectiveness of Lisinopril and increase kidney complications.",
    "Metformin combined with Alcohol may increase lactic acidosis risk.",
    "Sildenafil and Nitroglycerin together can dangerously lower blood pressure.",
    "Paracetamol and Alcohol together may increase liver damage risk.",
    "Aspirin and Clopidogrel together may cause excessive bleeding."
]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(docs)

# Build FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# User query
query = input("Ask a medical question: ")

# Encode query
query_embedding = model.encode([query])

# Search
k = 2

distances, indices = index.search(np.array(query_embedding), k)

print("\nTop Results:\n")

for idx in indices[0]:

    print(docs[idx])
    print()