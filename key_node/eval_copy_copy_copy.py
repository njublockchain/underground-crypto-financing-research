# %%
import os
import networkx
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

os.chdir("/home/lab0/coinwar")

# Define files and their respective importance keys in tuples (filename, node_importance_key)
files_and_keys = [
    # ("our_approach_predicted_critical_score", "normalized_score"),
    ("graph_with_normalized_centralities","degree_centrality"),
    ("graph_with_normalized_centralities","betweenness_centrality"),
    # ("graph_with_normalized_centralities","closeness_centrality"),
    ("graph_with_normalized_centralities","pagerank"),
    # ("graph_with_normalized_centralities","vote_rank_score_normalized"),
    ("gcn_predicted_critical_score","normalized_score"),
    ("gat_predicted_critical_score","normalized_score"),
    ("gin_predicted_critical_score","normalized_score"),
    ("sage_predicted_critical_score","normalized_score"),
    ("supergat_predicted_critical_score","normalized_score"),
    ("gatv2_predicted_critical_score","normalized_score"),
    ("film_predicted_critical_score","normalized_score"),
    ("our_approach_predicted_critical_score","normalized_score")
]

plt.figure(figsize=(10, 8))  # Define a figure for all plots

curves = []

for filename, node_importance_key in files_and_keys:
    G = networkx.read_gexf(f"/home/lab0/coinwar/key_node/{filename}.gexf")

    is_important_nodes_seq = []
    is_terrirism_nodes_seq = []

    for node, data in G.nodes(data=True):
        if data.get("is_terrirism") == 1:  # Use .get() to avoid KeyError
            is_terrirism_nodes_seq.append(1)
            if node_importance_key in data:
                is_important_nodes_seq.append(data[node_importance_key])
            else:
                is_important_nodes_seq.append(0)  # Default value if key is missing

    if filename == "graph_with_normalized_centralities":
        name: str = node_importance_key.split("_")[0]
    else:
        name = filename.split("_")[0]  + "+ K-Means" # Get the name of the model

    precision, recall, thresholds = precision_recall_curve(is_terrirism_nodes_seq, is_important_nodes_seq)
    curves.append((precision, recall, thresholds, name))
    # Plotting, but need to adjust recall and thresholds to match in length

# %%
plt.figure(figsize=(10, 6), dpi=80)
for i, curve in enumerate(curves):
    # if i in [0, 3, 9, 12]:
    #     continue
    print(len(curve[2]), len(curve[1]))
    plt.plot(curve[2], [rec for rec in curve[1][:len(curve[2])]], marker='.', label=f"{curve[3].capitalize() }")

plt.ylabel('Recall')  
plt.xlabel('Thresholds')
plt.title('Recall Curve')
plt.legend(title="File and Metric", loc="best")  # Add a legend with a title and automatic best location  # Optional: Add grid for better readability
plt.show()
# save_path = "/home/lab0/coinwar/key_node/precision_recall_curves.png"
# plt.savefig(save_path)

# %%