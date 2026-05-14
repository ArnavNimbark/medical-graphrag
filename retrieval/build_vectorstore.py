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

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(docs)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("Vector store created successfully.")