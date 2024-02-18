import networkx as nx
import pickle

# 加载.pkl文件中的图
with open('/home/lab0/coinwar/graphs/nx/all_1hop_pruned.pkl', 'rb') as f:
    G = pickle.load(f)

# 定义归一化函数
def normalize_dict_values(d):
    max_value = max(d.values())
    min_value = min(d.values())
    if max_value == min_value:  # 防止除以零
        return {k: 0 for k in d}
    return {k: (v - min_value) / (max_value - min_value) for k, v in d.items()}

# 计算中心性指标并进行归一化
degree_centrality = normalize_dict_values(nx.degree_centrality(G))
betweenness_centrality = normalize_dict_values(nx.betweenness_centrality(G))
closeness_centrality = normalize_dict_values(nx.closeness_centrality(G))
pagerank = normalize_dict_values(nx.pagerank(G))
# 计算k-shell（一般不归一化，但如果需要，也可以归一化）
# k_shell = nx.core_number(G)
# k_shell_normalized = normalize_dict_values(k_shell)

# 使用VoteRank算法计算节点的重要性
vote_rank = nx.voterank(G, number_of_nodes=len(G))
vote_rank_scores = {node: len(G) - rank for rank, node in enumerate(vote_rank)}  # 分数反映排名顺序
vote_rank_normalized = normalize_dict_values(vote_rank_scores)


# 将计算得到的归一化中心性指标作为节点属性加入图中
for node in G.nodes():
    G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)  # 使用.get()方法，如果节点不存在，则返回0
    G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
    G.nodes[node]['closeness_centrality'] = closeness_centrality.get(node, 0)
    G.nodes[node]['pagerank'] = pagerank.get(node, 0)
    # 对于vote_rank_score_normalized，使用.get()并提供默认值0
    G.nodes[node]['vote_rank_score_normalized'] = vote_rank_normalized.get(node, 0)

# 保存图为GEXF格式
nx.write_gexf(G, '/home/lab0/coinwar/key_node/graph_with_normalized_centralities.gexf')
