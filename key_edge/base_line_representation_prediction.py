import networkx as nx
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 加载图
def load_graph(graph_path):
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    return G

# IC模拟
def run_ic_simulation(G, initial_node, activation_prob=0.1):
    active = set([initial_node])
    newly_active = set([initial_node])
    
    while newly_active:
        current_newly_active = set()
        for node in newly_active:
            neighbors = set(G.successors(node)) - active  # 使用successors对于有向图
            for neighbor in neighbors:
                if random.random() < activation_prob:
                    active.add(neighbor)
                    current_newly_active.add(neighbor)
        newly_active = current_newly_active
    return len(active) - 1  # 不包括初始节点自身

# MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

def main():
    # 加载图和节点向量表示
    G = load_graph('/home/lab0/coinwar/graphs/nx/all_1hop_pruned.pkl')

    rep = pickle.load(open('/home/lab0/coinwar/key_node/baseline_film_with_for_critical_node.pkl',"rb"))  # 通过GAT变换后的图的节点向量表示
    final_representation = rep["final_representation"]
    index_to_node_id = rep["index_to_node_id"]

    # 执行IC模拟并准备数据
    node_infections = {node: run_ic_simulation(G, node) for node in G.nodes()}
    indices = [i for i, node in enumerate(G.nodes()) if node_infections[node] > 0]
    X = torch.tensor(final_representation[indices], dtype=torch.float)
    y = torch.tensor([node_infections[node] for node in G.nodes() if node_infections[node] > 0], dtype=torch.float)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义和训练MLP模型
    model = MLP(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    # 预测所有节点的感染能力
    with torch.no_grad():
        predictions = model(torch.tensor(final_representation, dtype=torch.float)).flatten()
    
    # 获取感染能力从高到低的节点索引和对应的分数
    sorted_indices = predictions.argsort(descending=True)
    sorted_scores = predictions[sorted_indices]
    
    # 归一化得分
    max_score = sorted_scores.max()
    min_score = sorted_scores.min()
    normalized_scores = (sorted_scores - min_score) / (max_score - min_score)

    # 使用索引到节点ID的映射转换最高score的节点索引到节点ID
    highest_score_node_id = index_to_node_id[sorted_indices[0].item()]

    # 转换Scores从高到低的节点索引到对应的节点ID
    sorted_node_ids = [index_to_node_id[int(i)] for i in sorted_indices]

    import json
    hamas = json.load(open("/home/lab0/coinwar/labels/hamas.json"))
    is_hamas = [1 if node in hamas else 0 for node in sorted_node_ids]

    # 显示结果
    # print(index_to_node_id)
    print("Score最高的节点ID:", highest_score_node_id)
    # print("Scores从高到低的节点ID:", sorted_node_ids)
    # print("Scores从高到低:", list(sorted_scores.numpy()))
    print(list(zip(sorted_node_ids, list(sorted_scores.numpy()), list(range(1, len(sorted_node_ids)+1)), is_hamas)))
    
    # is_hamas.rindex(1)
    print("last hamas node index", 1 + next(i for i in reversed(range(len(is_hamas))) if is_hamas[i] == 1))

    for node_id, score in zip(sorted_node_ids, normalized_scores):
        G.nodes[node_id]['normalized_score'] = score.item()

    # Exporting to GEXF
    nx.write_gexf(G, '/home/lab0/coinwar/key_node/film_predicted_critical_score.gexf')

# 假设你已经有了 final_representation 作为 NumPy 数组
# final_representation = ... # 你的节点向量表示
# graph_path = 'path_to_your_graph.pkl'  # 你的图文件路径

# 如果已经准备好 final_representation 和 graph_path，可以直接调用 main 函数
main()
