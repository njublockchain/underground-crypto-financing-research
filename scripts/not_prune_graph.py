import networkx
import pickle
import json

filename = "all_1hop"

G: networkx.MultiDiGraph = networkx.read_gexf(f"./legacy/{filename}.gexf")

print("节点：", G.number_of_nodes())
print("边：", G.number_of_edges())

# 计算每个节点的入度和出度
in_degrees = G.in_degree()
out_degrees = G.out_degree()

# 找出所有单向外围节点
# nodes_to_remove = [
#     node for node in G.nodes() if in_degrees[node] == 0 or out_degrees[node] == 0
# ]
# G.remove_nodes_from(nodes_to_remove)

# 找出无向外围节点
# nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree == 1]
# G.remove_nodes_from(nodes_to_remove)

relabels = {}
for node in G.nodes():
    if type(node) is not str:
        node_address = node.decode()
        relabels[node] = node_address
    if node.startswith("b'"):
        node_address = node.removeprefix("b'").removesuffix("'")
        relabels[node] = node_address

G = networkx.relabel_nodes(G, relabels)

values_out = {node: 0 for node in G.nodes()}
values_in = {node: 0 for node in G.nodes()}

times_out = {node: 0 for node in G.nodes()}
times_in = {node: 0 for node in G.nodes()}

for u, v, data in G.edges(data=True):
    value = data.get("value", 0)
    value = int(value) / 1000000
    values_out[u] += value  # 累加到出度
    values_in[v] += value  # 累加到入度

    blockNum = data.get("blockNum", 0)
    times_out[u] += blockNum
    times_in[v] += blockNum

ALL_HAMAS_ADDRS = json.load(open("./labels/hamas.json"))
for node in G.nodes():
    basic_feature = json.load(open(f"./tronscan/account_basic/{node}.json"))
    tfs_feature = json.load(open(f"./tronscan/account_usdt_tfs/{node}.json"))

    if node in ALL_HAMAS_ADDRS:
        is_terrirism = 1
    else:
        is_terrirism = 0

    networkx.set_node_attributes(
        G,
        {
            node: {
                "is_terrirism": is_terrirism,
                "degree": G.degree(node),
                "degree_in": G.in_degree(node),
                "degree_out": G.out_degree(node),
                "degree_diff_ratio": abs(
                    0
                    if G.degree(node) == 0
                    else (G.in_degree(node) - G.out_degree(node)) / G.degree(node)
                ),
                "degree_io_ratio": (
                    0
                    if G.in_degree(node) == 0
                    else G.out_degree(node) / G.in_degree(node)
                ),
                "value_in_sum": values_in.get(node, 0),
                "value_out_sum": values_out.get(node, 0),
                "value_in_avg": (
                    0
                    if not G.in_degree(node)
                    else values_in.get(node, 0) / G.in_degree(node)
                ),
                "value_out_avg": (
                    0
                    if not G.out_degree(node)
                    else values_out.get(node, 0) / G.out_degree(node)
                ),
                "value_sum": (values_in.get(node, 0) + values_out.get(node, 0)),
                "value_diff_ratio": abs(
                    0
                    if values_out.get(node, 0) + values_in.get(node, 0) == 0
                    else (values_out.get(node, 0) - values_in.get(node, 0))
                    / (values_out.get(node, 0) + values_in.get(node, 0))
                ),
                "value_io_ratio": (
                    0
                    if values_in.get(node, 0) == 0
                    else values_out.get(node, 0) / values_in.get(node, 0)
                ),
                "blocknum_in_avg": (
                    0
                    if not G.in_degree(node)
                    else times_in.get(node, 0) / G.in_degree(node)
                ),
                "blocknum_out_avg": (
                    0
                    if not G.out_degree(node)
                    else times_out.get(node, 0) / G.out_degree(node)
                ),
                "blocknum_avg": (
                    times_out.get(node, 0)
                    + times_in.get(node, 0) / (G.out_degree(node) + G.in_degree(node))
                ),
                "blocknum_diff_ratio": abs(times_out.get(node, 0) - times_in.get(node, 0))
                / (times_out.get(node, 0) + times_in.get(node, 0)),
                "usdt_transfer_in": tfs_feature.get("transferIn", 0),
                "usdt_transfer_out": tfs_feature.get("transferOut", 0),
                "usdt_transfer_all": tfs_feature.get("transferIn", 0)
                + tfs_feature.get("transferOut", 0),
                "usdt_transfer_diff_ratio": abs(
                    0
                    if tfs_feature.get("transferIn", 0)
                    + tfs_feature.get("transferOut", 0)
                    == 0
                    else (
                        tfs_feature.get("transferIn", 0)
                        - tfs_feature.get("transferOut", 0)
                    )
                    / (
                        tfs_feature.get("transferIn", 0)
                        + tfs_feature.get("transferOut", 0)
                    )
                ),
                "usdt_transfer_io_ratio": (
                    0
                    if tfs_feature.get("transferIn", 0) == 0
                    else tfs_feature.get("transferOut", 0)
                    / tfs_feature.get("transferIn", 0)
                ),
                "account_transaction_count_out": basic_feature.get(
                    "transactions_out", 0
                ),
                "account_transaction_count_in": basic_feature.get("transactions_in", 0),
                "account_transaction_count_all": basic_feature.get("transactions", 0),
                "account_transaction_diff_ratio": abs(
                    0
                    if basic_feature.get("transactions", 0) == 0
                    else (
                        basic_feature.get("transactions_in", 0)
                        - basic_feature.get("transactions_out", 0)
                    )
                    / basic_feature.get("transactions", 0)
                ),
                "account_transaction_io_ratio": (
                    0
                    if basic_feature.get("transactions_in", 0) == 0
                    else basic_feature.get("transactions_out", 0)
                    / basic_feature.get("transactions_in", 0)
                ),
                "date_created": basic_feature.get("date_created", 0),
                "name": basic_feature.get("name", ""),
                "feedbackRisk": (
                    1 if (basic_feature.get("feedbackRisk", False)) else 0
                ),
                "tags": ",".join(
                    [
                        basic_feature.get("redTag", ""),
                        basic_feature.get("greyTag", ""),
                        basic_feature.get("blueTag", ""),
                        basic_feature.get("publicTag", ""),
                    ]
                ),
            }
        },
    )

# 展示删除节点后的图
print("节点：", G.number_of_nodes())
print("边：", G.number_of_edges())

with open(f"./graphs/nx/{filename}_not_pruned.pkl", "wb") as f:
    pickle.dump(G, f)

with open(f"./graphs/gexf/{filename}_not_pruned.gexf", "wb") as f:
    networkx.write_gexf(G, f)
