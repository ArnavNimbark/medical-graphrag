import google.generativeai as genai

# -----------------------------------
# CONFIGURE GEMINI
# -----------------------------------

genai.configure(
    api_key="AIzaSyCbvbys4zHhz-OgtaqyEsG3f8SVDZdM5jQ"
)

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