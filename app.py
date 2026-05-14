import streamlit as st
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from streamlit_agraph import agraph, Node, Edge, Config
from retrieval.ai_reasoning import generate_reasoning

# ----------------------------
# PAGE TITLE
# ----------------------------

st.title("MedGraphRAG")
st.subheader("Drug Interaction Intelligence System")

# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv("data/drugs.csv")

# ----------------------------
# BUILD GRAPH
# ----------------------------

G = nx.MultiDiGraph()

for _, row in df.iterrows():

    G.add_edge(
        row["drug"],
        row["interacts_with"],
        effect=row["effect"]
    )

# ----------------------------
# TRADITIONAL RAG SETUP
# ----------------------------

docs = [
    "Warfarin and Aspirin together may increase bleeding risk.",
    "Ibuprofen may reduce the effectiveness of Lisinopril and increase kidney complications.",
    "Metformin combined with Alcohol may increase lactic acidosis risk.",
    "Sildenafil and Nitroglycerin together can dangerously lower blood pressure.",
    "Paracetamol and Alcohol together may increase liver damage risk.",
    "Aspirin and Clopidogrel together may cause excessive bleeding."
]

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(docs)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# ----------------------------
# USER INPUT
# ----------------------------

query = st.text_input("Ask a medical interaction question")

# ----------------------------
# PROCESS QUERY
# ----------------------------

if query:

    st.divider()

    # ----------------------------
    # TRADITIONAL RAG
    # ----------------------------

    st.subheader("Traditional RAG")

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        2
    )

    rag_results = []

    for idx in indices[0]:

        st.write(docs[idx])

        rag_results.append(docs[idx])

    # ----------------------------
    # GRAPHRAG
    # ----------------------------

    st.subheader("GraphRAG")

    graph_results = []

    found = False

    for node in G.nodes():

        if node.lower() in query.lower():

            found = True

            st.write(f"Detected Drug: {node}")

            for neighbor in G.neighbors(node):

                edge_data = G.get_edge_data(node, neighbor)

                for key in edge_data:

                    effect = edge_data[key]["effect"]

                    st.write(f"{node} → {neighbor}")
                    st.write(f"Effect: {effect}")

                    graph_results.append(
                        f"{node} interacts with {neighbor}. Effect: {effect}"
                    )

    if not found:

        st.write("No graph relationships found.")

    # --------------------------------
    # AI REASONING
    # --------------------------------

    st.subheader("AI Medical Reasoning")

    ai_response = generate_reasoning(
        query,
        rag_results,
        graph_results
    )

    st.success(ai_response)

    # ----------------------------
    # GRAPH VISUALIZATION
    # ----------------------------

    st.markdown("---")

    st.subheader("Graph Visualization")

    st.info("Interactive biomedical relationship graph")

    nodes = []

    edges = []

    added_nodes = set()

    for source, target, data in G.edges(data=True):

        # SOURCE NODE

        if source not in added_nodes:

            nodes.append(
                Node(
                    id=source,
                    label=source,
                    size=32,
                    color="#3B82F6",
                    shape="dot",
                    font={
                        "color": "#60A5FA",
                        "size": 18,
                        "face": "arial",
                        "strokeWidth": 2,
                        "strokeColor": "#000000"
                    }
                )
            )

            added_nodes.add(source)

        # TARGET NODE

        if target not in added_nodes:

            nodes.append(
                Node(
                    id=target,
                    label=target,
                    size=28,
                    color="#10B981",
                    shape="dot",
                    font={
                        "color": "#34D399",
                        "size": 18,
                        "face": "arial",
                        "strokeWidth": 2,
                        "strokeColor": "#000000"
                    }
                )
            )

            added_nodes.add(target)

        # EDGES

        edges.append(
            Edge(
                source=source,
                target=target,
                label=data["effect"][:20],

                color={
                    "color": "#FF4D4D",
                    "highlight": "#FF0000",
                    "hover": "#FF6666"
                },

                font={
                    "color": "#FF5555",
                    "size": 14,
                    "face": "arial",
                    "strokeWidth": 3,
                    "strokeColor": "#000000"
                }
            )
        )

    # ----------------------------
    # GRAPH CONFIG
    # ----------------------------

    config = Config(
        width="100%",
        height=1200,
        directed=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True,

        physics={
            "enabled": True,
            "solver": "forceAtlas2Based",

            "forceAtlas2Based": {
                "gravitationalConstant": -300,
                "centralGravity": 0.003,
                "springLength": 350,
                "springConstant": 0.02,
                "damping": 0.4,
                "avoidOverlap": 1
            },

            "minVelocity": 0.75,
            "timestep": 0.35
        }
    )

    # ----------------------------
    # DISPLAY GRAPH
    # ----------------------------

    agraph(
        nodes=nodes,
        edges=edges,
        config=config,
    )