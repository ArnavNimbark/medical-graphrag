from datasets import load_dataset
import os
from tqdm import tqdm

# Create folder
os.makedirs("medical_docs", exist_ok=True)

# Load PubMed dataset - using a working alternative
dataset = load_dataset("ccdv/pubmed-summarization", split="train[:1000]")

# Save abstracts as text files
for i, item in tqdm(enumerate(dataset), total=1000):

    article = item.get("article", "")

    if len(article.strip()) < 500:
        continue

    with open(f"medical_docs/doc_{i}.txt", "w", encoding="utf-8") as f:
        f.write(article)

print("Done!")