import networkx as nx
import pickle
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

# 定义加载图的函数
def load_graph(path):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G

# 定义执行随机游走的函数
def random_walk(G, start_node, L):
    walk = [start_node]
    for _ in range(1, L):
        neighbors = list(G.successors(walk[-1]))
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        walk.append(next_node)
    return walk

# 定义聚合随机游走中的边属性的函数
def agg_w(args):
    G, walks = args
    results = []
    for walk in walks:
        edge_features = np.array([G[walk[i]][walk[i+1]].get('value', 0) for i in range(len(walk)-1)])
        edge_blocknums = np.array([G[walk[i]][walk[i+1]].get('blockNum', 0) for i in range(len(walk)-1)])
        if len(edge_features) == 0:
            results.append(np.zeros(2))
        else:
            features_mean = np.mean(np.stack((edge_features, edge_blocknums), axis=1), axis=0)
            results.append(features_mean)
    return results

# 编码器模型
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, edge_attr, S_i, S_j):
        x = torch.cat((edge_attr, S_i, S_j), dim=-1)
        return F.relu(self.fc(x))

def process_edge_data(G, edge, k, L):
    i, j = edge
    walks_i = [random_walk(G, i, L) for _ in range(k)]
    walks_j = [random_walk(G, j, L) for _ in range(k)]
    return agg_w((G, walks_i)), agg_w((G, walks_j))

def main():
    path = '/home/lab0/coinwar/graphs/nx/all_1hop_pruned.pkl'  # 更换为你的.pkl文件路径
    G = load_graph(path)
    k, L = 4, 2
    
    # 初始化编码器模型和优化器
    encoder = Encoder(6, 4)  # 假设输入维度6，输出维度4
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 准备多进程计算随机游走和特征聚合
    edges = list(G.edges())
    with Pool(os.cpu_count()) as pool:
        results = pool.starmap(process_edge_data, [(G, edge, k, L) for edge in edges])
    
    Si_list, Sj_list = zip(*results)
    Si_array = np.mean(np.array(Si_list), axis=1)
    Sj_array = np.mean(np.array(Sj_list), axis=1)

    edge_attr_array = np.array([[G[u][v].get('value', 0), G[u][v].get('blockNum', 0)] for u, v in G.edges()], dtype=float)
    edge_attr = torch.tensor(edge_attr_array, dtype=torch.float)

    Si_tensor = torch.tensor(Si_array, dtype=torch.float)
    Sj_tensor = torch.tensor(Sj_array, dtype=torch.float)

    target_embeddings = torch.rand(len(G.edges()), 4)  # 随机目标嵌入，仅用于示例
    
    # 模拟训练过程
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = encoder(edge_attr, Si_tensor, Sj_tensor)
        loss = criterion(outputs, target_embeddings)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    # 使用轮廓系数选择最佳聚类数
    embeddings = outputs.detach().numpy()
    best_n_clusters, best_score = -1, -1
    for n_clusters in range(2, 10):
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        if score > best_score:
            best_n_clusters, best_score = n_clusters, score

    print(f"Best number of clusters: {best_n_clusters}, Silhouette Score: {best_score}")

    # 更新图属性并保存为GEXF
    clustering = AgglomerativeClustering(n_clusters=best_n_clusters).fit(embeddings)
    for (u, v), cluster_label in zip(G.edges(), clustering.labels_):
        G[u][v]['cluster'] = cluster_label
    nx.write_gexf(G, "clustered_graph.gexf")

    # PCA降维并可视化
    pca = PCA(n_components=2)
    edge_embeddings_2d = pca.fit_transform(embeddings)
    plt.scatter(edge_embeddings_2d[:, 0], edge_embeddings_2d[:, 1], c=clustering.labels_)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()
