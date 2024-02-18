# %%
import networkx as nx
import pickle
import numpy as np
from node2vec import Node2Vec
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 加载.pkl文件
with open('/home/lab0/coinwar/graphs/nx/all_1hop_pruned.pkl', 'rb') as f:
    G = pickle.load(f)

# 配置Node2Vec参数
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

# 训练Node2Vec模型
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 生成边的嵌入表示，这次使用哈达玛积
def get_edge_embedding_hadamard(edge):
    node_embedding_1 = model.wv[str(edge[0])]
    node_embedding_2 = model.wv[str(edge[1])]
    return np.multiply(node_embedding_1, node_embedding_2)

edges_embeddings = {edge: get_edge_embedding_hadamard(edge) for edge in set(G.edges())}

# %%
# 准备嵌入向量用于聚类
embeddings_list = np.array(list(edges_embeddings.values()))

# 将嵌入向量转换为距离矩阵
distance_matrix = pdist(embeddings_list, 'euclidean')

# 执行AHC聚类
Z = linkage(distance_matrix, 'ward')

# 选择最优聚类数量基于轮廓系数
silhouette_scores = []
for n_clusters in range(2, min(30, len(G.edges()) + 1)):  # 测试2到10个聚类
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    score = silhouette_score(embeddings_list, labels, metric='euclidean')
    silhouette_scores.append(score)
    print(f'Silhouette score for {n_clusters} clusters: {score}')

# 找到轮廓系数最高的聚类数量
optimal_clusters = np.argmax(silhouette_scores) + 2  # 加2因为范围是从2开始的
optimal_labels = fcluster(Z, optimal_clusters, criterion='maxclust')

print(f'The Size of Optimal number of clusters: {optimal_labels}')
print(G.number_of_edges())

# %%
# 将聚类标签添加到边属性中
for i, edge in enumerate(G.edges):
    # 为每条边设置聚类标签
    G.edges[edge]['cluster'] = int(optimal_labels[i])


# 保存聚类后的图对象
with open('/home/lab0/coinwar/key_edge/clustered_graph_edge_node2vec_clusters.pkl', 'wb') as f:
    pickle.dump(G, f)

# 生成二维聚类图
plt.figure(figsize=(10, 8))
for i, label in enumerate(optimal_labels):
    plt.scatter(embeddings_list[i, 0], embeddings_list[i, 1], label=str(label))
plt.title('2D Cluster Plot of Edges')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()
