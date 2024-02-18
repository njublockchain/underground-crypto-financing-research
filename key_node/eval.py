import networkx
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score

filename = "gcn_predicted_critical_score" # CHANGEME
G = networkx.read_gexf(f"/home/lab0/coinwar/key_node/{filename}.gexf")

node_importance_key = "normalized_score" 
node_importance_theshold = 0.8

is_important_nodes_seq = []
is_terrirism_nodes_seq = []

for node, data in G.nodes(data=True):
    if data[node_importance_key] > node_importance_theshold:
        is_important_nodes_seq.append(1)
    else:
        is_important_nodes_seq.append(0)
    
    if data["is_terrirism"] == 1:
        is_terrirism_nodes_seq.append(1)
    else:
        is_terrirism_nodes_seq.append(0)

acc = accuracy_score(is_terrirism_nodes_seq, is_important_nodes_seq)
recall = recall_score(is_terrirism_nodes_seq, is_important_nodes_seq)
precision = precision_score(is_terrirism_nodes_seq, is_important_nodes_seq)
auc = roc_auc_score(is_terrirism_nodes_seq, is_important_nodes_seq)
f1 = f1_score(is_terrirism_nodes_seq, is_important_nodes_seq)

print(f"Node Accuracy: {acc}, Recall: {recall}, Precision: {precision}, AUC: {auc}, F1: {f1}")


# edge_importance_key = "importance"
# edge_importance_theshold = 0.8

# is_important_edges_seq = []
# is_terrirism_edges_seq = []

# for u, v, data in G.edges(data=True):
#     if data[edge_importance_key] > edge_importance_theshold:
#         is_important_edges_seq.append(1)
#     else:
#         is_important_edges_seq.append(0)
    
#     # get data of u and v
#     u_data = G.nodes[u]
#     v_data = G.nodes[v]

#     if u_data["is_terrirism"] == 1 or v_data["is_terrirism"] == 1:
#         is_terrirism_edges_seq.append(1)
#     else:
#         is_terrirism_edges_seq.append(0)

# acc = accuracy_score(is_terrirism_edges_seq, is_important_edges_seq)
# recall = recall_score(is_terrirism_edges_seq, is_important_edges_seq)
# precision = precision_score(is_terrirism_edges_seq, is_important_edges_seq)
# auc = roc_auc_score(is_terrirism_edges_seq, is_important_edges_seq)
# f1 = f1_score(is_terrirism_edges_seq, is_important_edges_seq)

# print(f"Edge Accuracy: {acc}, Recall: {recall}, Precision: {precision}, AUC: {auc}, F1: {f1}")
