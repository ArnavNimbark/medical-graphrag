import os

import google.generativeai as genai
from dotenv import load_dotenv

# -----------------------------------
# CONFIGURE GEMINI
# -----------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

genai.configure(api_key=api_key)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = genai.GenerativeModel("gemini-flash-latest")

# -----------------------------------
# AI REASONING FUNCTION
# -----------------------------------

def generate_reasoning(query, rag_results=None, graph_results=None, mode="Hybrid GraphRAG"):
    rag_results = rag_results or []
    graph_results = graph_results or []

    prompt = f"""

You are a medical AI assistant.

Pipeline Mode:
{mode}

User Question:
{query}

Traditional RAG Results:
{rag_results}

Graph Relationship Results:
{graph_results}

Generate a concise medical explanation using only the evidence available
for this pipeline mode. If evidence is limited, say so clearly.

Include:
1. Main interaction risk
2. Why the interaction happens
3. Clinical significance

Keep response under 120 words.

"""

    response = model.generate_content(prompt)

    return response.text
