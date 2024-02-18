# %%
import os
import networkx

os.chdir("/home/lab0/coinwar")
filename = "updated_graph_with_ahc"  # CHANGEME
G: networkx.MultiDiGraph = networkx.read_gexf(f"key_node/{filename}.gexf")

predefined_9_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]
colors = []
for node, data in G.nodes(data=True):
    cluster_id = data["ahc_cluster"]
    colors.append(predefined_9_colors[cluster_id])

from matplotlib import pyplot as plt
# plt.figure(num=None, figsize=(20, 20), dpi=80)
# plt.axis('off')
# fig = plt.figure(1)

# networkx.draw(
#     G, pos=networkx.kamada_kawai_layout(G), node_color=colors, with_labels=True
# )
# %%
P = networkx.nx_pydot.to_pydot(G)

output_raw_dot = P.to_string()
# Or, save it as a DOT-file:
P.write_raw("output_raw.dot")

P.write_png("output.png")
# %%