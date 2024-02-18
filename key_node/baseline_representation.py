from sklearnex import patch_sklearn
patch_sklearn()
import torch
import torch.nn as nn
import hdbscan
import pickle
from torch_geometric.data import Data
import torch.optim as optim
from torch.nn import Sequential
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from torch_geometric.nn import GATConv
from torch_geometric.utils.convert import from_networkx
from sklearn.cluster import KMeans
import networkx as nx
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import softmax
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from torch_geometric.nn import GATConv, GINConv, SAGEConv, GCNConv, SuperGATConv, GATv2Conv, FiLMConv
import torch.nn.functional as F
import torch
import pickle
from torch_geometric.data import Data

# 假设pkl文件路径
pkl_path = '/home/lab0/coinwar/graphs/nx/all_1hop_pruned.pkl'


# 读取pkl文件
with open(pkl_path, 'rb') as f:
    G = pickle.load(f)

# 创建节点ID到连续整数索引的映射
node_id_to_index = {node_id: idx for idx, node_id in enumerate(G.nodes())}

# 使用映射来构建edge_index
edge_index = torch.tensor(
    [[node_id_to_index[u], node_id_to_index[v]] for u, v in G.edges()], 
    dtype=torch.long
).t().contiguous()

# 定义节点特征的标题列表
node_feature_titles = [
    "degree", "degree_in", "degree_out", "degree_diff_ratio", "degree_io_ratio",
    "value_in_sum", "value_out_sum", "value_in_avg", "value_out_avg", "value_sum",
    "value_diff_ratio", "value_io_ratio", "blocknum_in_avg", "blocknum_out_avg",
    "blocknum_avg", "blocknum_diff_ratio", "usdt_transfer_in", "usdt_transfer_out",
    "usdt_transfer_all", "usdt_transfer_diff_ratio", "usdt_transfer_io_ratio",
    "account_transaction_count_out", "account_transaction_count_in", "account_transaction_count_all",
    "account_transaction_diff_ratio", "account_transaction_io_ratio"
]

# 提取节点特征
node_features = []
for node in G.nodes():
    node_attrs = G.nodes[node]
    features = []
    for title in node_feature_titles:
        # 对于未定义的属性，使用0作为默认值
        features.append(node_attrs.get(title, 0))
    node_features.append(features)

# 对节点特征进行归一化
node_features_nor = normalize(node_features, axis=0)  # 归一化每个特征维度

# 将列表转换为张量
node_features_tensor = torch.tensor(node_features_nor, dtype=torch.float)

# 提取边属性
edge_attributes = []
for u, v, attrs in G.edges(data=True):
    edge_attributes.append([attrs.get('value', 0), attrs.get('blockNum', 0)])

# 将边属性列表转换为张量
edge_attr = torch.tensor(edge_attributes, dtype=torch.float)


# 创建Data对象
data = Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_attr)


# 定义GAT模型
class CustomGATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomGATConv, self).__init__()
        self.conv = GATConv(in_channels, out_channels, heads=1)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

# 定义GIN模型
class CustomGINConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomGINConv, self).__init__()
        self.conv = GINConv(nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)))

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

# 定义GraphSAGE模型
class CustomSAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomSAGEConv, self).__init__()
        self.conv = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

# 定义GCN模型
class CustomGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomGCNConv, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

# 定义SuperGAT模型
class CustomSuperGATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomSuperGATConv, self).__init__()
        self.conv = SuperGATConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
    
# 定义GATv2模型
class CustomGATv2Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomGATv2Conv, self).__init__()
        self.conv = GATv2Conv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

# 定义FilM模型
class CustomFiLMConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomFiLMConv, self).__init__()
        self.conv = FiLMConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

# 定义模型
gat_model = CustomGATConv(node_features_tensor.size(1), 64)
gin_model = CustomGINConv(node_features_tensor.size(1), 64)
sage_model = CustomSAGEConv(node_features_tensor.size(1), 64)
gcn_model = CustomGCNConv(node_features_tensor.size(1), 64)
supergat_model = CustomSuperGATConv(node_features_tensor.size(1), 64)
gatv2_model = CustomGATv2Conv(node_features_tensor.size(1), 64)
film_model = CustomFiLMConv(node_features_tensor.size(1), 64)

# 对每个模型进行前向传播
gat_output = gat_model(node_features_tensor, edge_index)
gin_output = gin_model(node_features_tensor, edge_index)
sage_output = sage_model(node_features_tensor, edge_index)
gcn_output = gcn_model(node_features_tensor, edge_index)
supergat_output = supergat_model(node_features_tensor, edge_index)
gatv2_output = gatv2_model(node_features_tensor, edge_index)
film_output = film_model(node_features_tensor, edge_index)


final_representation = film_output

# 如果最终表示是一维的，则将其重塑为二维（[num_nodes, num_features]）
if final_representation.dim() == 1:
    final_representation = final_representation.view(-1, 1)

# 将最终表示从 GPU 移动到 CPU，并转换为 NumPy 数组
final_representation_detached = final_representation.detach().cpu().numpy()

# 准备输出数据
output_data = {
    'final_representation': final_representation_detached,
    'index_to_node_id': {idx: node_id for node_id, idx in node_id_to_index.items()}
}

# 将输出数据保存到文件中
with open('/home/lab0/coinwar/key_node/baseline_film_with_for_critical_node.pkl', 'wb') as f:
    pickle.dump(output_data, f)
