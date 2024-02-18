import networkx as nx
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from multiprocessing import Pool

# 加载图
def load_graph(path):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G

# 执行随机游走
def random_walk(G, start_node, L):
    walk = [start_node]
    for _ in range(1, L):
        neighbors = list(G.successors(walk[-1]))
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        walk.append(next_node)
    return walk

# 聚合随机游走中的边属性
def agg_w(walk, G):
    edge_features = np.array([G[walk[i]][walk[i+1]].get('value', 0) for i in range(len(walk)-1)])
    edge_blocknums = np.array([G[walk[i]][walk[i+1]].get('blockNum', 0) for i in range(len(walk)-1)])
    if len(edge_features) == 0:  # 如果没有边特征，返回零向量
        return np.zeros(2)
    features_mean = np.mean(np.stack((edge_features, edge_blocknums), axis=1), axis=0)
    return features_mean

# 编码器模型
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, edge_attr, S_i, S_j):
        x = torch.cat((edge_attr, S_i, S_j), dim=-1)
        return F.relu(self.fc(x))

def main():
    path = '/home/lab0/coinwar/graphs/nx/all_1hop_pruned.pkl'
    G = load_graph(path)
    k, L = 5, 10
    
    encoder = Encoder(6, 4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    
    # 提取真实的边属性
    # 提取真实的边属性并直接转换为numpy数组
    edge_attr_array = np.array([[G[u][v].get('value', 1), G[u][v].get('blockNum', 0)] for u, v in G.edges()], dtype=float)

    # 将numpy数组转换为torch.tensor
    edge_attr = torch.tensor(edge_attr_array, dtype=torch.float)
    
    target_embeddings = torch.rand(len(G.edges()), 4)  # 随机生成的目标嵌入，仅用于示例

    # 模拟训练过程
    for epoch in range(5):  # 为了简化示例，这里只训练10轮
        Si_list, Sj_list = [], []
        for edge in G.edges():
            i, j = edge
            walks_i = [random_walk(G, i, L) for _ in range(k)]
            walks_j = [random_walk(G, j, L) for _ in range(k)]
            Si = np.mean([agg_w(walk, G) for walk in walks_i], axis=0)
            Sj = np.mean([agg_w(walk, G) for walk in walks_j], axis=0)
            Si_list.append(Si)
            Sj_list.append(Sj)
        
        # 使用numpy的stack函数将列表转换为numpy数组
        Si_array = np.stack(Si_list)
        Sj_array = np.stack(Sj_list)

        # 然后，将numpy数组转换为torch.tensor
        Si_tensor = torch.tensor(Si_array, dtype=torch.float)
        Sj_tensor = torch.tensor(Sj_array, dtype=torch.float)

        optimizer.zero_grad()
        E_uv = encoder(edge_attr, Si_tensor, Sj_tensor)
        loss = criterion(E_uv, target_embeddings)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    # 训练完成后的操作
    print("Training complete.")

    # 使用轮廓系数选择最佳聚类数
    silhouette_scores = []
    for n_clusters in range(2, 10):  # 测试的聚类数量范围
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering.fit_predict(E_uv.detach().numpy())
        score = silhouette_score(E_uv.detach().numpy(), cluster_labels)
        silhouette_scores.append(score)
        
    best_n_clusters = np.argmax(silhouette_scores) + 2  # +2 因为范围是从2开始的
    best_clustering = AgglomerativeClustering(n_clusters=best_n_clusters).fit(E_uv.detach().numpy())
    best_cluster_labels = best_clustering.labels_
    
    # 更新图属性并保存为GEXF
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['cluster'] = int(best_cluster_labels[i])
    nx.write_gexf(G, "/home/lab0/coinwar/key_edge/edge_clustered_graph.gexf")
    
    # PCA降维并可视化
    pca = PCA(n_components=2)
    edge_embeddings_2d = pca.fit_transform(E_uv.detach().numpy())
    plt.scatter(edge_embeddings_2d[:, 0], edge_embeddings_2d[:, 1], c=best_cluster_labels)
    plt.title('Edge Embeddings 2D Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()