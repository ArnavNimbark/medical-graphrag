import pandas as pd
import networkx as nx
from pyvis.network import Network

# Create graph
G = nx.MultiDiGraph()

# Load CSV
df = pd.read_csv("data/drugs.csv")

# Add nodes and edges
for _, row in df.iterrows():

    drug = row["drug"]
    interacts = row["interacts_with"]
    effect = row["effect"]

    G.add_node(drug, type="drug")
    G.add_node(interacts, type="drug")

    G.add_edge(
        drug,
        interacts,
        title=effect
    )

# Create interactive network
net = Network(
    height="750px",
    width="100%",
    bgcolor="#222222",
    font_color="white"
)

net.from_nx(G)

# Save HTML
net.write_html("drug_graph.html")

print("Graph visualization created.")