import torch
import torch.nn as nn
import torch.optim as optim  # 仍然导入，但只用于初始学习率
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

# --- 超参数 ---
num_nodes = 10
max_byzantine_nodes = 2
learning_rate = 0.01
batch_size = 64
num_epochs = 500
trim_parameter = 2
connectivity = 0.7

# --- 设备选择 (优先使用 cuda) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- 递减学习率序列 ---
def lr_schedule(epoch):
    return learning_rate / (1 + 0.1 * math.log(epoch+1))

# --- 数据加载 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)

trainloaders = []
subset_size = len(trainset) // num_nodes
for i in range(num_nodes):
    indices = list(range(i * subset_size, (i + 1) * subset_size))
    subset = torch.utils.data.Subset(trainset, indices)
    trainloaders.append(torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                               shuffle=True))

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

# --- 模型定义 ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(1024, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 1024)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 初始化模型 ---
models = [SimpleCNN().to(device) for _ in range(num_nodes)]
# optimizers = [optim.SGD(model.parameters(), lr=learning_rate) for model in models]  # 不再需要
criterion = nn.CrossEntropyLoss()

# --- 邻接矩阵 ---
def create_weighted_doubly_stochastic_matrix(num_agents, prob):
    """
    1. Create an Erdős-Rényi random graph.
    2. Assign random positive weights to edges.
    3. Transform into a doubly stochastic matrix (Sinkhorn-Knopp).
    """
    G = nx.erdos_renyi_graph(num_agents, prob)
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 1)
    return A

adj_matrix = create_weighted_doubly_stochastic_matrix(num_nodes, connectivity)
print(f"Adjacency Matrix:\n{adj_matrix}")

# --- 拜占庭节点 ---
byzantine_indices = random.sample(range(num_nodes),
                                  random.randint(0, max_byzantine_nodes))
print(f"Byzantine node indices: {byzantine_indices}")

# --- 筛选函数 ---
def trimmed_mean_screen(params_list, trim_param):
    num_params = len(params_list[0])
    aggregated_params = []

    for param_idx in range(num_params):
        param_values = torch.stack([params[param_idx] for params in params_list], dim=0)
        sorted_values, _ = torch.sort(param_values, dim=0)
        trimmed_values = sorted_values[trim_param:len(params_list) - trim_param]
        aggregated_param = torch.mean(trimmed_values, dim=0)
        aggregated_params.append(aggregated_param)

    return aggregated_params

all_epoch_losses = [[] for _ in range(num_nodes)]

# --- 训练循环 ---
for epoch in range(num_epochs):
    current_lr = lr_schedule(epoch)
    # current_lr = 0.01
    epoch_losses = [[] for _ in range(num_nodes)]

    for batch_idx in range(len(trainloaders[0])):

        # 0. 每个节点计算本地梯度
        local_gradients = []
        for node_idx in range(num_nodes):
            model = models[node_idx]
            data, target = next(iter(trainloaders[node_idx]))
            data, target = data.to(device), target.to(device)

            model.zero_grad()  # 使用模型的 zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            epoch_losses[node_idx].append(loss.item())

            grads = [param.grad.data.clone() for param in model.parameters()]
            local_gradients.append(grads)

        # 1. 广播模型参数
        all_params = []
        for node_idx in range(num_nodes):
            model_params = [param.data.clone() for param in models[node_idx].parameters()]
            all_params.append(model_params)
            if node_idx in byzantine_indices:
                for i in range(len(model_params)):
                    all_params[-1][i] = torch.randn_like(all_params[-1][i])

        # 2. 接收并筛选
        filtered_params = []
        for node_idx in range(num_nodes):
            neighbor_indices = np.where(adj_matrix[node_idx])[0]
            # print(neighbor_indices)
            neighbor_params = [all_params[i] for i in neighbor_indices]
            aggregated_params = trimmed_mean_screen(neighbor_params, trim_parameter)
            filtered_params.append(aggregated_params)

        # 3. 更新模型 
        for node_idx in range(num_nodes):
            if node_idx not in byzantine_indices:

                for param, agg_param, local_grad in zip(models[node_idx].parameters(),
                                                        filtered_params[node_idx],
                                                        local_gradients[node_idx]):
                    param.data = agg_param - current_lr * local_grad  # 直接更新参数

    for node_idx in range(num_nodes):
        all_epoch_losses[node_idx].append(np.mean(epoch_losses[node_idx]))

    # --- 每个 epoch 结束时，打印损失 ---
    print(f"Epoch {epoch+1}, LR: {current_lr:.4f}")
    for node_idx in range(num_nodes):
        avg_loss = np.mean(epoch_losses[node_idx])
        print(f"  Node {node_idx}: Avg. Loss = {avg_loss:.4f}")

    # --- 每 10 个 epoch 进行一次评估 ---
    if (epoch + 1) % 10 == 0:
        all_accuracies = []
        for node_idx in range(num_nodes):
            if node_idx not in byzantine_indices:
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in testloader:
                        data, target = data.to(device), target.to(device)
                        outputs = models[node_idx](data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                accuracy = 100 * correct / total
                all_accuracies.append(accuracy)
                print(f"  Evaluation (Epoch {epoch+1}): Node {node_idx} Accuracy = {accuracy:.2f}%")

        if all_accuracies:
            print(f"  Average Accuracy (non-Byzantine nodes): {np.mean(all_accuracies):.2f}%")

print('Finished Training. Evaluating...')
accuracies = []
for node_idx in range(num_nodes):
    if node_idx not in byzantine_indices:
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                outputs = models[node_idx](data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f"Node {node_idx}: Accuracy = {accuracy:.2f}%")
    else:
        accuracies.append(0)  # Placeholder for Byzantine nodes

# --- 绘图 ---
plt.figure(figsize=(12, 8))

# 1. 绘制每个节点的准确率
plt.subplot(2, 2, 1)  # 2x2 网格，第 1 个子图
plt.bar(range(num_nodes), accuracies, color=['blue' if i not in byzantine_indices else 'red' for i in range(num_nodes)])
plt.xlabel("Node Index")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy of Each Node")
plt.ylim(0, 100)
for i in byzantine_indices:
    plt.text(i, accuracies[i] + 2, "Byzantine", ha='center', color='red')

# 2. 绘制每个节点的损失曲线
plt.subplot(2, 2, 2)  # 2x2 网格，第 2 个子图
for node_idx in range(num_nodes):
    if node_idx not in byzantine_indices:
        plt.plot(all_epoch_losses[node_idx], label=f"Node {node_idx}")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Loss Curves (Non-Byzantine Nodes)")
plt.legend()

# 3. 绘制每个节点的损失曲线，纵轴采用对数刻度
plt.subplot(2, 2, 3)  # 2x2 网格，第 3 个子图
for node_idx in range(num_nodes):
    if node_idx not in byzantine_indices:
        plt.plot(all_epoch_losses[node_idx], label=f"Node {node_idx}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves (Log Scale, Non-Byzantine Nodes)")
plt.yscale("log")  # 设置 y 轴为对数刻度
plt.legend()


# 4. 可视化邻接矩阵
plt.subplot(2, 2, 4)  # 2x2 网格，第 4 个子图
plt.imshow(adj_matrix, cmap="Blues")
plt.title("Adjacency Matrix")
plt.colorbar()  # 添加颜色条

plt.tight_layout()  # 自动调整子图间距
plt.show()
plt.savefig("bridge.png")