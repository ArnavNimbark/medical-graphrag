import pandas as pd
import networkx as nx

# Create graph
G = nx.MultiDiGraph()

# Load data
df = pd.read_csv("data/drugs.csv")

# Build graph
for _, row in df.iterrows():

    G.add_edge(
        row["drug"],
        row["interacts_with"],
        effect=row["effect"]
    )

# Ask for drug
query_drug = input("Enter a drug name: ")

print(f"\nInteractions for {query_drug}:\n")

# Find neighbors
if query_drug in G:

    for neighbor in G.neighbors(query_drug):

        edge_data = G.get_edge_data(query_drug, neighbor)

        for key in edge_data:

            effect = edge_data[key]["effect"]

            print(f"{query_drug} → {neighbor}")
            print(f"Effect: {effect}\n")

else:
    print("Drug not found.")