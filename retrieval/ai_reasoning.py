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

def generate_reasoning(query, rag_results, graph_results):

    prompt = f"""

You are a medical AI assistant.

User Question:
{query}

Traditional RAG Results:
{rag_results}

Graph Relationship Results:
{graph_results}

Using both semantic retrieval and graph relationships,
generate a concise medical explanation.

Include:
1. Main interaction risk
2. Why the interaction happens
3. Clinical significance

Keep response under 120 words.

"""

    response = model.generate_content(prompt)

    return response.text
