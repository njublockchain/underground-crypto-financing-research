import os
import networkx
import json

# filename = "updated_graph_with_hdbscan"
filename = "updated_graph_with_ahc"
# filename = "updated_graph_with_ahc_pruned_not_temporal"
# cluster_key = "kdd_cluster" 
cluster_key = "ahc_cluster"

G = networkx.read_gexf(f"key_node/{filename}.gexf")
unique_clusters = {}

for node, data in G.nodes(data=True):
    cluster_id = str(data[cluster_key])
    if cluster_id not in unique_clusters:
        unique_clusters[cluster_id] = [node]
    else:
        unique_clusters[cluster_id].append(node)

print(len(unique_clusters))

# for node, data in G.nodes(data=True):
#     print(data.keys())
#     break

# calc min/max/avg for each clusters
all_cluster_stats = {}
all_real_labelled_key_node_stats_in_cluster = {}
all_other_node_stats_in_cluster = {}
for cluster_id, cluster in unique_clusters.items():
    # print(cluster)
    cluster_stats = {
        "degree": {}, # min/max/avg
        "degree_in": {}, # min/max/avg
        "degree_out": {}, # min/max/avg
        "degree_diff_ratio": {}, # min/max/avg
        "degree_io_ratio": {}, # min/max/avg
        "value_in_sum": {}, # min/max/avg
        "value_out_sum": {}, # min/max/avg
        "value_in_avg": {}, # min/max/avg
        "value_out_avg": {}, # min/max/avg
        "value_sum": {}, # min/max/avg
        "value_diff_ratio": {}, # min/max/avg
        "value_io_ratio": {}, # min/max/avg
        "blocknum_in_avg": {}, # min/max/avg
        "blocknum_out_avg": {}, # min/max/avg
        "blocknum_diff_ratio": {}, # min/max/avg
        "usdt_transfer_in": {}, # min/max/avg
        "usdt_transfer_out": {}, # min/max/avg
        "usdt_transfer_diff_ratio": {}, # min/max/avg
        "usdt_transfer_io_ratio": {}, # min/max/avg
        "account_transaction_count_out": {}, # min/max/avg
        "account_transaction_count_in": {}, # min/max/avg
        "account_transaction_count_all": {}, # min/max/avg
        "account_transaction_diff_ratio": {}, # min/max/avg
        "account_transaction_io_ratio": {}, # min/max/avg
        "date_created": {}, # min/max/avg
    }

    real_labelled_key_node_stats_in_cluster = {
        "degree": {}, # min/max/avg
        "degree_in": {}, # min/max/avg
        "degree_out": {}, # min/max/avg
        "degree_diff_ratio": {}, # min/max/avg
        "degree_io_ratio": {}, # min/max/avg
        "value_in_sum": {}, # min/max/avg
        "value_out_sum": {}, # min/max/avg
        "value_in_avg": {}, # min/max/avg
        "value_out_avg": {}, # min/max/avg
        "value_sum": {}, # min/max/avg
        "value_diff_ratio": {}, # min/max/avg
        "value_io_ratio": {}, # min/max/avg
        "blocknum_in_avg": {}, # min/max/avg
        "blocknum_out_avg": {}, # min/max/avg
        "blocknum_diff_ratio": {}, # min/max/avg
        "usdt_transfer_in": {}, # min/max/avg
        "usdt_transfer_out": {}, # min/max/avg
        "usdt_transfer_diff_ratio": {}, # min/max/avg
        "usdt_transfer_io_ratio": {}, # min/max/avg
        "account_transaction_count_out": {}, # min/max/avg
        "account_transaction_count_in": {}, # min/max/avg
        "account_transaction_count_all": {}, # min/max/avg
        "account_transaction_diff_ratio": {}, # min/max/avg
        "account_transaction_io_ratio": {}, # min/max/avg
        "date_created": {}, # min/max/avg
    }


    other_node_stats_in_cluster = {
        "degree": {}, # min/max/avg
        "degree_in": {}, # min/max/avg
        "degree_out": {}, # min/max/avg
        "degree_diff_ratio": {}, # min/max/avg
        "degree_io_ratio": {}, # min/max/avg
        "value_in_sum": {}, # min/max/avg
        "value_out_sum": {}, # min/max/avg
        "value_in_avg": {}, # min/max/avg
        "value_out_avg": {}, # min/max/avg
        "value_sum": {}, # min/max/avg
        "value_diff_ratio": {}, # min/max/avg
        "value_io_ratio": {}, # min/max/avg
        "blocknum_in_avg": {}, # min/max/avg
        "blocknum_out_avg": {}, # min/max/avg
        "blocknum_diff_ratio": {}, # min/max/avg
        "usdt_transfer_in": {}, # min/max/avg
        "usdt_transfer_out": {}, # min/max/avg
        "usdt_transfer_diff_ratio": {}, # min/max/avg
        "usdt_transfer_io_ratio": {}, # min/max/avg
        "account_transaction_count_out": {}, # min/max/avg
        "account_transaction_count_in": {}, # min/max/avg
        "account_transaction_count_all": {}, # min/max/avg
        "account_transaction_diff_ratio": {}, # min/max/avg
        "account_transaction_io_ratio": {}, # min/max/avg
        "date_created": {}, # min/max/avg
    }


    cluster_cache = {}
    key_nodes_cache = {}
    others_cache = {}
    for node in cluster:
        node_data = G.nodes[node]

        # save values into cluster cache
        for key in cluster_stats.keys():
            if key in cluster_cache:
                cluster_cache[key] += [node_data[key]]
            else:
                cluster_cache[key] = [node_data[key]]

        if node_data["is_terrirism"] > 0: # is real_labelled_key_node
            for key in real_labelled_key_node_stats_in_cluster.keys():
                if key in key_nodes_cache:
                    key_nodes_cache[key] += [node_data[key]]
                else:
                    key_nodes_cache[key] = [node_data[key]]
        else:
            for key in other_node_stats_in_cluster.keys():
                if key in others_cache:
                    others_cache[key] += [node_data[key]]
                else:
                    others_cache[key] = [node_data[key]]

    print(cluster_cache)
    # calculate min/max/avg for each cluster
    for key, value in cluster_cache.items():
        cluster_stats[key]["min"] = min(value)
        cluster_stats[key]["max"] = max(value)
        cluster_stats[key]["avg"] = sum(value) / len(value)
    
    for key, value in key_nodes_cache.items():
        real_labelled_key_node_stats_in_cluster[key]["min"] = min(value)
        real_labelled_key_node_stats_in_cluster[key]["max"] = max(value)
        real_labelled_key_node_stats_in_cluster[key]["avg"] = sum(value) / len(value)
    
    for key, value in others_cache.items():
        other_node_stats_in_cluster[key]["min"] = min(value)
        other_node_stats_in_cluster[key]["max"] = max(value)
        other_node_stats_in_cluster[key]["avg"] = sum(value) / len(value)
    
    # print(cluster_stats)
    # print(real_labelled_key_node_stats_in_cluster)
    # print(other_node_stats_in_cluster)
    all_cluster_stats[cluster_id] = cluster_stats
    all_real_labelled_key_node_stats_in_cluster[cluster_id] = real_labelled_key_node_stats_in_cluster
    all_other_node_stats_in_cluster[cluster_id] = other_node_stats_in_cluster

os.makedirs("key_node/cluster_stats", exist_ok=True)

json.dump(all_cluster_stats, open(f"cluster_stats/{filename}_all_cluster_stats.json", "w"), indent=4)
json.dump(all_real_labelled_key_node_stats_in_cluster, open(f"cluster_stats/{filename}_all_real_labelled_key_node_stats_in_cluster.json", "w"), indent=4)
json.dump(all_other_node_stats_in_cluster, open(f"cluster_stats/{filename}_all_other_node_stats_in_cluster.json", "w"), indent=4)
