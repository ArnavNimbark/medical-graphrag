# MedGraphRAG

MedGraphRAG is a Streamlit prototype for comparing medical question-answering pipelines over drug interaction and PubMed-style medical text data.

The app now supports:

- **LLM-only baseline**: asks Gemini directly with no retrieved evidence.
- **Basic RAG baseline**: searches local PubMed text files from `medical_docs/` with SentenceTransformers + FAISS.
- **GraphRAG pipeline**: combines semantic retrieval with graph evidence from `data/drugs.csv`, with optional TigerGraph GraphRAG API integration.
- **Metrics comparison**: estimates latency, tokens, cost, evidence count, and evaluation-based answer quality score for each pipeline.
- **Explainability**: shows retrieved evidence, graph reasoning paths, interaction risk, confidence, and clinical recommendation panels.
- **Interactive graph visualization**: expands detected drug relationships with PyVis.

This is an educational/research prototype, not a clinical decision support system.

## Project Structure

```text
medical-graphrag/
  app.py                         # Main Streamlit comparison app
  evaluation.py                  # Evaluation scoring helpers
  tigergraph_client.py           # Optional TigerGraph GraphRAG API client
  download_pubmed.py             # Downloads PubMed-style articles into medical_docs/
  data/
    drugs.csv                    # Small curated drug interaction graph
    drug_docs.txt                # Legacy demo corpus
    evaluation_questions.csv     # Small benchmark set with expected answer terms
  graph/
    build_graph.py               # Prints graph nodes/edges
    visualize_graph.py           # Generates drug_graph.html with PyVis
  retrieval/
    ai_reasoning.py              # Gemini reasoning helper
    document_store.py            # PubMed document loading, FAISS indexing, search
    build_vectorstore.py         # Legacy in-memory FAISS demo
    query_vectorstore.py         # Legacy CLI semantic search demo
    query_graph.py               # Legacy CLI graph lookup demo
    gemini_reasoning.py          # Standalone Gemini test/demo
  medical_docs/                  # Downloaded PubMed-style .txt articles
```

## Current Status

Done:

- Downloaded PubMed-style dataset into `medical_docs/`.
- App uses those local text files for Basic RAG retrieval.
- App compares LLM-only, Basic RAG, and GraphRAG-style answers.
- Local graph evidence comes from `data/drugs.csv`.
- Gemini API key is read from `.env`.
- Runtime metrics are shown in the UI.
- Answer quality now uses `data/evaluation_questions.csv` instead of a pure heuristic.
- `medical_docs/` has been loaded into the TigerGraph `MedicalGraphRAG` graph as raw `Document`/`Content` vertices.
- Docker files are present for containerized app runs.

Still improving:

- TigerGraph GraphRAG extraction/rebuild can take a while after ingestion because it chunks, embeds, extracts entities, and builds graph communities.
- The local graph is still a small curated CSV graph.
- Token and cost metrics are estimates, not provider billing records.
- The evaluation set is intentionally small; expand it for stronger research claims.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_gemini_key_here
```

Optional TigerGraph GraphRAG API settings:

```env
TIGERGRAPH_GRAPHRAG_URL=http://localhost:8000
TIGERGRAPH_GRAPH_NAME=MedicalGraphRAG
TIGERGRAPH_USERNAME=tigergraph
TIGERGRAPH_PASSWORD=tigergraph
```

If `TIGERGRAPH_GRAPHRAG_URL` is not set, the app automatically uses the local CSV graph fallback.

## Dataset

Download the PubMed-style text dataset:

```powershell
pip install datasets
python download_pubmed.py
```

This creates `medical_docs/*.txt`. The Streamlit app loads up to 975 text files and builds a cached FAISS index from them on first run.

## Run

```powershell
streamlit run app.py
```

Useful test questions:

- `Why is Warfarin dangerous with Aspirin?`
- `Can Ibuprofen interact with Lisinopril?`
- `What happens with Sildenafil and Nitroglycerin?`
- `What complications occur when aspirin and ibuprofen are used together in cardiac patients?`

## Docker

Build the app image:

```powershell
docker build -t medical-graphrag -f dockerfile .
```

Run it:

```powershell
docker run --env-file .env -p 8501:8501 medical-graphrag
```

Then open:

```text
http://localhost:8501
```

## Metrics

The comparison table reports:

- `Latency (s)`: measured wall-clock time for each pipeline.
- `Estimated Tokens`: rough character-based token estimate.
- `Estimated Cost ($)`: rough cost estimate using a configurable placeholder rate.
- `Evidence Items`: number of retrieved text snippets and graph facts.
- `Answer Quality Score`: keyword coverage against the closest row in `data/evaluation_questions.csv`, plus an evidence bonus.

These metrics are good for early project comparison, but not a substitute for a proper benchmark set.

## TigerGraph GraphRAG Ingestion

Current local TigerGraph status:

- Graph: `MedicalGraphRAG`
- Raw document load: completed for 975 PubMed-style files
- Loaded objects: 975 `Document` vertices, 975 `Content` vertices, 975 `HAS_CONTENT` edges
- GraphRAG rebuild: run from the UI/API after ingestion to create chunks, embeddings, entities, relationships, and communities

The app will call the TigerGraph API only when `TIGERGRAPH_GRAPHRAG_URL` is set. Otherwise it uses the local CSV graph fallback.

## Safety Note

Generated drug interaction explanations can be incomplete or wrong. Use trusted medical references and qualified clinical judgment for real healthcare decisions.
