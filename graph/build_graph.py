import pandas as pd
import networkx as nx

# Create graph
G = nx.MultiDiGraph()

# Load CSV
df = pd.read_csv("data/drugs.csv")

# Add nodes and edges
for _, row in df.iterrows():

    drug = row["drug"]
    interacts = row["interacts_with"]
    effect = row["effect"]

    # Add nodes
    G.add_node(drug, type="drug")
    G.add_node(interacts, type="drug")

    # Add edge
    G.add_edge(
        drug,
        interacts,
        relation="interacts_with",
        effect=effect
    )

print("=== NODES ===")
print(G.nodes(data=True))

print("\\n=== EDGES ===")
print(G.edges(data=True))