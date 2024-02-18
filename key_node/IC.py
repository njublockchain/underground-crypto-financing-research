import pickle
import csv
import random

# 加载图数据
def load_graph(graph_path):
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    return graph

# IC模拟
def ic_simulation(graph, activation_prob=0.1):
    infected_count = {node: 0 for node in graph.nodes()}  # 初始化每个节点感染数量为0

    for node in graph.nodes():
        active = set([node])
        newly_active = set([node])

        while newly_active:
            current_newly_active = set()
            for current_node in newly_active:
                # 获取当前节点的邻居节点
                neighbors = set(graph.successors(current_node)) - active
                for neighbor in neighbors:
                    if random.random() < activation_prob:
                        active.add(neighbor)
                        current_newly_active.add(neighbor)
            newly_active = current_newly_active

        # 记录每个节点感染其他节点的数量
        infected_count[node] = len(active) - 1  # 不包括初始节点自身

    return infected_count

# 加载图数据
graph = load_graph('/home/lab0/coinwar/graphs/nx/all_1hop_pruned.pkl')

# 执行IC模拟
infected_counts = ic_simulation(graph)

# 打印每个节点感染其他节点的数量
for node, count in infected_counts.items():
    print(f"Node {node}: Infected {count} nodes")

# 输出每个节点及其感染其他节点数量的list
node_infection_counts_list = [(node, count) for node, count in infected_counts.items()]
print(node_infection_counts_list)


# 打开或创建 CSV 文件以写入数据
with open('/home/lab0/coinwar/key_node/node_infection_counts.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Node', 'Infected_Node_Count'])  # 写入表头

    # 循环遍历每个节点及其感染其他节点数量
    for node, count in infected_counts.items():
        if count != 0:  # 仅输出感染其他节点数量不为零的节点
            writer.writerow([node, count])  # 写入节点和感染其他节点数量
