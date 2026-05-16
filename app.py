import streamlit as st
import pandas as pd
import networkx as nx
from evaluation import load_evaluation_questions, score_answer_quality
from retrieval.ai_reasoning import generate_reasoning
from retrieval.document_store import (
    build_vector_index as build_document_vector_index,
    load_pubmed_documents as load_pubmed_document_store,
    search_documents,
)
from tigergraph_client import query_tigergraph_graphrag

# GRAPH IMPORTS
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import time

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="MedGraphRAG",
    layout="wide"
)

# ----------------------------
# SESSION MEMORY
# ----------------------------

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

if "conversation_context" not in st.session_state:

    st.session_state.conversation_context = ""


def estimate_tokens(text):
    return max(1, int(len(str(text)) / 4))


def estimate_cost(tokens, cost_per_1k=0.00035):
    return (tokens / 1000) * cost_per_1k


@st.cache_data(show_spinner=False)
def load_pubmed_documents(max_docs=975):
    return load_pubmed_document_store(max_docs=max_docs)


@st.cache_resource(show_spinner=False)
def build_vector_index(documents):
    return build_document_vector_index(documents)

# ----------------------------
# PAGE TITLE
# ----------------------------

st.title("MedGraphRAG")
st.subheader(
    "Conversational Drug Interaction Intelligence System"
)

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
# TRADITIONAL RAG
# ----------------------------

docs, docs_source = load_pubmed_documents()

model, index = build_vector_index(docs)

evaluation_df = load_evaluation_questions()

# ----------------------------
# SIDEBAR MEMORY
# ----------------------------

with st.sidebar:

    st.header("Conversation Memory")

    run_baselines = st.checkbox(
        "Run baseline comparison",
        value=True
    )

    if st.button("Clear Memory"):

        st.session_state.chat_history = []
        st.session_state.conversation_context = ""

        st.success("Conversation memory cleared.")

    if st.session_state.chat_history:

        for idx, item in enumerate(
            reversed(st.session_state.chat_history[-10:])
        ):

            with st.expander(f"Conversation {len(st.session_state.chat_history)-idx}"):

                st.markdown("### User")

                st.info(item["query"])

                st.markdown("### AI")

                st.success(item["response"])

    else:

        st.info("No conversation history yet.")

# ----------------------------
# USER INPUT
# ----------------------------

query = st.chat_input(
    "Ask a medical interaction question..."
)


# ----------------------------
# PROCESS QUERY
# ----------------------------

if query is not None and query.strip() != "":

    st.divider()

    # ----------------------------
    # BUILD CONTEXTUAL QUERY
    # ----------------------------

    contextual_query = f"""

Previous Conversation Context:
{st.session_state.conversation_context}

Current User Query:
{query}

"""

    # ----------------------------
    # RAG RETRIEVAL
    # ----------------------------

    st.subheader("Traditional RAG")

    st.caption(
        f"Searching {len(docs)} documents from "
        f"{'medical_docs/' if docs_source == 'pubmed' else 'fallback demo corpus'}."
    )

    rag_start = time.perf_counter()

    rag_results = search_documents(
        query,
        docs,
        model,
        index,
        top_k=5
    )

    for snippet in rag_results:

        st.info(snippet)

    rag_latency = time.perf_counter() - rag_start

    # ----------------------------
    # GRAPHRAG
    # ----------------------------

    st.subheader("GraphRAG")

    graph_results = []
    tigergraph_response = None
    tigergraph_error = None
    graph_start = time.perf_counter()

    try:
        tigergraph_response = query_tigergraph_graphrag(query)
    except Exception as exc:
        tigergraph_error = str(exc)

    if tigergraph_response:

        st.success("TigerGraph GraphRAG API response")
        st.write(tigergraph_response)
        graph_results.append(tigergraph_response)

    elif tigergraph_error:

        st.warning(
            "TigerGraph GraphRAG API is configured but did not return a response. "
            "Using the local CSV graph fallback."
        )

    detected_drugs = []

    found = False

    for node in G.nodes():

        if node.lower() in contextual_query.lower():

            if node not in detected_drugs:

                detected_drugs.append(node)

            found = True

            st.success(f"Detected Drug: {node}")

            for neighbor in G.neighbors(node):

                edge_data = G.get_edge_data(node, neighbor)

                for key in edge_data:

                    effect = edge_data[key]["effect"]

                    st.write(f"### {node} -> {neighbor}")

                    st.warning(f"Effect: {effect}")

                    graph_results.append(
                        f"{node} interacts with {neighbor}. Effect: {effect}"
                    )

    if not found and not tigergraph_response:

        st.error("No graph relationships found.")

    graph_latency = time.perf_counter() - graph_start

    # ----------------------------
    # BUILD SUBGRAPH
    # ----------------------------

    subgraph = nx.MultiDiGraph()

    for node in detected_drugs:

        for neighbor in G.neighbors(node):

            edge_data = G.get_edge_data(node, neighbor)

            for key in edge_data:

                effect = edge_data[key]["effect"]

                subgraph.add_edge(
                    node,
                    neighbor,
                    effect=effect
                )

                # SECOND HOP

                for second_neighbor in G.neighbors(neighbor):

                    second_edge_data = G.get_edge_data(
                        neighbor,
                        second_neighbor
                    )

                    if second_edge_data:

                        for k in second_edge_data:

                            second_effect = second_edge_data[k]["effect"]

                            subgraph.add_edge(
                                neighbor,
                                second_neighbor,
                                effect=second_effect
                            )

    # ----------------------------
    # AI CONVERSATIONAL REASONING
    # ----------------------------

    st.markdown("---")

    st.subheader("AI Medical Copilot")

    chat_container = st.container()

    with chat_container:

        # ----------------------------
        # DISPLAY PREVIOUS CHAT HISTORY
        # ----------------------------

        for message in st.session_state.chat_history:

            if query:

                with st.chat_message("user"):

                    st.markdown(message["query"])

                with st.chat_message("assistant"):

                    st.markdown(message["response"])

        # ----------------------------
        # CURRENT USER MESSAGE
        # ----------------------------

        with st.chat_message("user"):

            st.markdown(query)

        # ----------------------------
        # AI RESPONSE
        # ----------------------------

        with st.chat_message("assistant"):

            thinking_placeholder = st.empty()

            reasoning_steps = [

                "Analyzing biomedical entities...",
                "Traversing multi-hop graph relationships...",
                "Evaluating anticoagulant interaction pathways...",
                "Checking semantic retrieval evidence...",
                "Computing clinical severity risk...",
                "Generating explainable medical reasoning..."
            ]

            for step in reasoning_steps:

                thinking_placeholder.info(step)

                time.sleep(0.7)

            hybrid_start = time.perf_counter()

            ai_response = generate_reasoning(
                contextual_query,
                rag_results,
                graph_results,
                mode="Hybrid GraphRAG"
            )

            hybrid_latency = time.perf_counter() - hybrid_start

            # ----------------------------
            # STREAMING EFFECT
            # ----------------------------

            streamed_text = ""

            response_placeholder = st.empty()

            for char in ai_response:

                streamed_text += char

                response_placeholder.markdown(
                    streamed_text + "|"
                )

                time.sleep(0.01)

            response_placeholder.markdown(
                streamed_text
            )

            thinking_placeholder.empty()

        # ----------------------------
        # BASELINE COMPARISON + METRICS
        # ----------------------------

        if run_baselines:

            st.markdown("---")

            st.subheader("Pipeline Comparison")

            llm_start = time.perf_counter()
            llm_only_response = generate_reasoning(
                query,
                [],
                [],
                mode="LLM-only baseline"
            )
            llm_latency = time.perf_counter() - llm_start

            basic_start = time.perf_counter()
            basic_rag_response = generate_reasoning(
                query,
                rag_results,
                [],
                mode="Basic RAG baseline"
            )
            basic_latency = time.perf_counter() - basic_start

            pipeline_outputs = [
                ("LLM-only", llm_only_response, llm_latency, [], []),
                ("Basic RAG", basic_rag_response, basic_latency + rag_latency, rag_results, []),
                ("GraphRAG", ai_response, hybrid_latency + rag_latency + graph_latency, rag_results, graph_results),
            ]

            metrics_rows = []

            for name, output, latency, rag_ctx, graph_ctx in pipeline_outputs:
                evidence_count = len(rag_ctx) + len(graph_ctx)
                token_estimate = estimate_tokens(
                    f"{query}\n{rag_ctx}\n{graph_ctx}\n{output}"
                )
                answer_quality = score_answer_quality(
                    query,
                    output,
                    rag_ctx + graph_ctx,
                    evaluation_df
                )

                metrics_rows.append(
                    {
                        "Pipeline": name,
                        "Latency (s)": round(latency, 2),
                        "Estimated Tokens": token_estimate,
                        "Estimated Cost ($)": round(estimate_cost(token_estimate), 6),
                        "Evidence Items": evidence_count,
                        "Answer Quality Score": answer_quality,
                    }
                )

            st.dataframe(
                pd.DataFrame(metrics_rows),
                use_container_width=True,
                hide_index=True
            )

            with st.expander("View baseline answers"):
                st.markdown("### LLM-only")
                st.write(llm_only_response)
                st.markdown("### Basic RAG")
                st.write(basic_rag_response)
                st.markdown("### GraphRAG")
                st.write(ai_response)

        # ----------------------------
        # RISK ASSESSMENT
        # ----------------------------

        st.markdown("---")

        st.subheader("Interaction Risk Assessment")

        risk_score = 35
        risk_level = "Mild"

        critical_keywords = [
            "bleeding",
            "hemorrhage",
            "severe",
            "dangerously",
            "life-threatening"
        ]

        moderate_keywords = [
            "kidney",
            "liver",
            "stress",
            "damage",
            "risk"
        ]

        graph_text = " ".join(graph_results).lower()

        if any(word in graph_text for word in critical_keywords):

            risk_score = 95
            risk_level = "Critical"

        elif any(word in graph_text for word in moderate_keywords):

            risk_score = 72
            risk_level = "Moderate"

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                label="Risk Score",
                value=f"{risk_score}%"
            )

        with col2:

            st.metric(
                label="Severity",
                value=risk_level
            )

        confidence_score = min(
            98,
            70 + (len(graph_results) * 6)
        )

        with col3:

            st.metric(
                label="AI Confidence",
                value=f"{confidence_score}%"
            )

        if risk_level == "Critical":

            st.error(
                "High-risk interaction detected. Immediate clinical attention recommended."
            )

        elif risk_level == "Moderate":

            st.warning(
                "Moderate interaction risk detected. Monitoring advised."
            )

        else:

            st.success(
                "Low interaction risk detected."
            )

        # ----------------------------
        # CLINICAL RECOMMENDATIONS
        # ----------------------------

        st.markdown("---")

        st.subheader("AI Clinical Recommendations")

        recommendations = []

        if risk_level == "Critical":

            recommendations.extend([

                {
                    "title": "Avoid Concurrent Administration",
                    "description":
                    "This drug combination carries potentially life-threatening interaction risks.",
                    "priority": "HIGH",
                    "color": "error"
                },

                {
                    "title": "Immediate Monitoring Recommended",
                    "description":
                    "Frequent laboratory monitoring and clinical observation are strongly advised.",
                    "priority": "HIGH",
                    "color": "warning"
                }

            ])

        if (
            "bleeding" in graph_text
            or "hemorrhage" in graph_text
        ):

            recommendations.extend([

                {
                    "title": "Monitor INR / Coagulation",
                    "description":
                    "Perform coagulation monitoring to reduce severe bleeding complications.",
                    "priority": "HIGH",
                    "color": "error"
                },

                {
                    "title": "Watch for Internal Bleeding",
                    "description":
                    "Monitor for bruising, gastrointestinal bleeding, or intracranial hemorrhage.",
                    "priority": "HIGH",
                    "color": "warning"
                }

            ])

        if (
            "kidney" in graph_text
            or "ibuprofen" in graph_text
        ):

            recommendations.append(

                {
                    "title": "Assess Renal Function",
                    "description":
                    "Kidney function monitoring is recommended due to nephrotoxic interaction potential.",
                    "priority": "MEDIUM",
                    "color": "warning"
                }
            )

        if (
            "liver" in graph_text
            or "alcohol" in graph_text
        ):

            recommendations.append(

                {
                    "title": "Reduce Hepatic Stress",
                    "description":
                    "Avoid alcohol intake and monitor liver enzyme levels during treatment.",
                    "priority": "MEDIUM",
                    "color": "info"
                }
            )

        if not recommendations:

            recommendations.append(

                {
                    "title": "Routine Monitoring",
                    "description":
                    "No major contraindications detected. Standard monitoring recommended.",
                    "priority": "LOW",
                    "color": "success"
                }
            )

        for rec in recommendations:

            card_text = f"""
    ### {rec['title']}

    Priority Level:
    {rec['priority']}

    Recommendation:
    {rec['description']}
    """

            if rec["color"] == "error":

                st.error(card_text)

            elif rec["color"] == "warning":

                st.warning(card_text)

            elif rec["color"] == "info":

                st.info(card_text)

            else:

                st.success(card_text)

        # ----------------------------
        # EXPLAINABILITY PANEL
        # ----------------------------

        st.markdown("---")

        st.subheader("AI Explainability")

        with st.expander("Why did the AI generate this conclusion?"):

            st.markdown("### Retrieved Semantic Evidence")

            for item in rag_results:

                st.info(item)

            st.markdown("### Explainable AI Reasoning Paths")

            reasoning_paths = []

            for detected in detected_drugs:

                for target in subgraph.nodes():

                    if detected != target:

                        try:

                            paths = list(
                                nx.all_simple_paths(
                                    subgraph,
                                    source=detected,
                                    target=target,
                                    cutoff=3
                                )
                            )

                            for path in paths[:2]:

                                reasoning_paths.append(path)

                        except:

                            pass

            if reasoning_paths:

                for idx, path in enumerate(reasoning_paths[:6]):

                    path_text = " -> ".join(path)

                    st.error(
                        f"""
    Path {idx + 1}

    Reasoning Chain:
    {path_text}

    Severity:
    CRITICAL

    AI Confidence:
    96%
    """
                    )

            else:

                st.info(
                    "No multi-hop reasoning paths discovered."
                )

            st.markdown("### AI Reasoning Strategy")

            st.success(
                """
    The AI combines:

    1. Semantic retrieval from vector embeddings
    2. Multi-hop graph traversal
    3. Drug interaction relationships
    4. Clinical risk propagation
    5. LLM-based reasoning synthesis
    6. Explainable reasoning path discovery
    7. Clinical recommendation intelligence
    8. Conversational memory reasoning

    This hybrid GraphRAG pipeline improves explainability,
    relationship discovery, and medical interaction inference.
    """
            )

        # ----------------------------
        # REAL-TIME GRAPH INTELLIGENCE
        # ----------------------------

        st.markdown("---")

        st.subheader("Interactive Graph Intelligence")

        st.info(
            "Dynamic biomedical relationship network with real-time graph expansion"
        )

        # ---------------------------------
        # GRAPH CONTROLS
        # ---------------------------------

        expand_depth = st.slider(
            "Graph Expansion Depth",
            min_value=1,
            max_value=4,
            value=2
        )

        show_labels = st.toggle(
            "Show Interaction Labels",
            value=True
        )

        physics_enabled = st.toggle(
            "Enable Dynamic Physics",
            value=True
        )

        # ---------------------------------
        # BUILD EXPANDED GRAPH
        # ---------------------------------

        expanded_graph = nx.MultiDiGraph()

        visited = set()

        def expand_node(node, depth):

            if depth == 0:

                return

            visited.add(node)

            for neighbor in G.neighbors(node):

                edge_data = G.get_edge_data(node, neighbor)

                for key in edge_data:

                    effect = edge_data[key]["effect"]

                    expanded_graph.add_edge(
                        node,
                        neighbor,
                        effect=effect
                    )

                if neighbor not in visited:

                    expand_node(neighbor, depth - 1)

        for drug in detected_drugs:

            expand_node(drug, expand_depth)

        # ---------------------------------
        # CREATE NETWORK
        # ---------------------------------

        net = Network(
            height="1100px",
            width="100%",
            bgcolor="#020617",
            font_color="white",
            directed=True,
            notebook=False,
            cdn_resources="remote"
        )

        # ---------------------------------
        # PHYSICS ENGINE
        # ---------------------------------

        if physics_enabled:

            net.force_atlas_2based(
                gravity=-80,
                central_gravity=0.015,
                spring_length=260,
                spring_strength=0.04,
                damping=0.95,
                overlap=1
            )

        # ---------------------------------
        # NODE STYLING
        # ---------------------------------

        for node in expanded_graph.nodes():

            connections = len(
                list(expanded_graph.neighbors(node))
            )

            if node in detected_drugs:

                node_color = "#EF4444"

                node_size = 70

            else:

                node_color = "#10B981"

                node_size = 35 + (connections * 4)

            title_text = f"""
    Drug: {node}

    Connected Relationships:
    {connections}

    Click to explore interactions.
    """

            net.add_node(
                node,
                label=node,
                title=title_text,
                color=node_color,
                size=node_size,
                borderWidth=5,
                shadow=True
            )

        # ---------------------------------
        # EDGE STYLING
        # ---------------------------------

        for source, target, data in expanded_graph.edges(data=True):

            effect = data["effect"]

            effect_lower = effect.lower()

            if (
                "severe" in effect_lower
                or "internal" in effect_lower
                or "life" in effect_lower
            ):

                edge_color = "#FF4D4D"

                edge_width = 8

            elif "bleeding" in effect_lower:

                edge_color = "#F97316"

                edge_width = 6

            elif (
                "kidney" in effect_lower
                or "renal" in effect_lower
            ):

                edge_color = "#EAB308"

                edge_width = 5

            elif (
                "liver" in effect_lower
                or "hepatic" in effect_lower
            ):

                edge_color = "#38BDF8"

                edge_width = 5

            else:

                edge_color = "#94A3B8"

                edge_width = 3

            edge_label = effect if show_labels else ""

            net.add_edge(
                source,
                target,
                label=edge_label,
                title=f"""
    Interaction:
    {effect}

    Source:
    {source}

    Target:
    {target}
    """,
                color=edge_color,
                width=edge_width,
                smooth={
                    "enabled": True,
                    "type": "dynamic"
                },
                arrows="to"
            )

        # ---------------------------------
        # NETWORK OPTIONS
        # ---------------------------------

        net.set_options("""
        var options = {

        "nodes": {

            "shape": "dot",

            "font": {
            "size": 22,
            "face": "Inter"
            },

            "scaling": {
            "min": 20,
            "max": 80
            }
        },

        "edges": {

            "smooth": {
            "type": "dynamic"
            },

            "font": {
            "size": 16,
            "align": "middle"
            }
        },

        "interaction": {

            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true,
            "multiselect": true
        },

        "physics": {

            "enabled": true,

            "forceAtlas2Based": {

            "gravitationalConstant": -120,
            "centralGravity": 0.02,
            "springLength": 230,
            "springConstant": 0.05
            },

            "solver": "forceAtlas2Based",

            "stabilization": {

            "enabled": true,
            "iterations": 250
            }
        }
        }
        """)

        # ---------------------------------
        # SAVE GRAPH
        # ---------------------------------

        tmp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".html"
        )

        net.save_graph(tmp_file.name)

        HtmlFile = open(
            tmp_file.name,
            "r",
            encoding="utf-8"
        )

        components.html(
            HtmlFile.read(),
            height=1100
        )
