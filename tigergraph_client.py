import os

import requests


def query_tigergraph_graphrag(question):
    api_url = os.getenv("TIGERGRAPH_GRAPHRAG_URL", "").rstrip("/")
    graph_name = os.getenv("TIGERGRAPH_GRAPH_NAME", "MedicalGraphRAG")
    username = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
    password = os.getenv("TIGERGRAPH_PASSWORD", "tigergraph")

    if not api_url:
        return None

    response = requests.post(
        f"{api_url}/{graph_name}/query",
        json={"query": question, "rag_method": "hybrid"},
        auth=(username, password),
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("natural_language_response") or str(data)
