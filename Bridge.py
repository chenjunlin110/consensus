import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import os
from datetime import datetime

# --- Hyperparameters ---
num_nodes = 10
max_byzantine_nodes = 2
learning_rate = 0.01
batch_size = 64
num_epochs = 400
trim_parameter = 2  # For BRIDGE-T and BRIDGE-B
connectivity = 0.8
seed = 42  # For reproducibility

# --- Set random seeds for reproducibility ---
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# --- Device selection ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Learning rate schedule ---
def lr_schedule(epoch):
    return learning_rate / (1 + 0.1 * math.log(epoch + 1))


# --- Data loading and preprocessing ---
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    # Distribute data among nodes
    trainloaders = []
    subset_size = len(trainset) // num_nodes
    for i in range(num_nodes):
        indices = list(range(i * subset_size, (i + 1) * subset_size))
        subset = torch.utils.data.Subset(trainset, indices)
        trainloaders.append(torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                                        shuffle=True))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)

    return trainloaders, testloader


# --- Model definition ---
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

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 1024)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# --- Network and Byzantine node setup ---
def create_adjacency_matrix(num_nodes, connectivity, seed):
    graph = nx.erdos_renyi_graph(num_nodes, connectivity, seed=seed)

    # Ensure the graph is connected
    while not nx.is_connected(graph):
        graph = nx.erdos_renyi_graph(num_nodes, connectivity, seed=seed)
        seed += 1

    adj_matrix = nx.to_numpy_array(graph)
    np.fill_diagonal(adj_matrix, 1)  # Include self-loops
    return adj_matrix, graph


def select_byzantine_nodes(num_nodes, max_byzantine):
    byzantine_indices = random.sample(range(num_nodes), max_byzantine)
    print(f"Byzantine node indices: {byzantine_indices}")
    return byzantine_indices


# --- Screening functions ---
def trimmed_mean_screen(params_list, trim_param):
    """BRIDGE-T: Coordinate-wise trimmed mean"""
    num_params = len(params_list[0])
    aggregated_params = []

    for param_idx in range(num_params):
        # Get original shape for reshaping later
        original_shape = params_list[0][param_idx].shape

        # Reshape based on dimensionality
        if len(original_shape) > 1:  # For multi-dimensional tensors
            param_values = torch.stack([p[param_idx].reshape(-1) for p in params_list], dim=0)
        else:  # For vectors
            param_values = torch.stack([p[param_idx] for p in params_list], dim=0)

        # Sort values along the first dimension (nodes)
        sorted_values, _ = torch.sort(param_values, dim=0)

        # Get trimmed values
        if len(params_list) > 2 * trim_param:  # Ensure we have enough values after trimming
            trimmed_values = sorted_values[trim_param:len(params_list) - trim_param]
        else:
            # If not enough values, use middle value
            trimmed_values = sorted_values[len(params_list) // 2].unsqueeze(0)

        # Calculate mean
        aggregated_param = torch.mean(trimmed_values, dim=0)

        # Reshape back to original shape and add to aggregated parameters
        aggregated_params.append(aggregated_param.reshape(original_shape))

    return aggregated_params


def median_screen(params_list):
    """BRIDGE-M: Coordinate-wise median"""
    num_params = len(params_list[0])
    aggregated_params = []

    for param_idx in range(num_params):
        original_shape = params_list[0][param_idx].shape

        # Handle different tensor dimensions
        if len(original_shape) > 1:  # For multi-dimensional tensors
            param_values = torch.stack([p[param_idx].reshape(-1) for p in params_list], dim=0)
        else:  # For vectors
            param_values = torch.stack([p[param_idx] for p in params_list], dim=0)

        # Get median values - torch.median returns (values, indices)
        median_values = torch.median(param_values, dim=0)[0]

        # Reshape back to original shape
        aggregated_params.append(median_values.reshape(original_shape))

    return aggregated_params


def krum_screen(params_list, num_byzantine):
    """BRIDGE-K: Krum screening"""
    num_neighbors = len(params_list)
    num_to_select = num_neighbors - num_byzantine - 2

    # If the condition for Krum is not met, return the first parameter set
    if num_to_select <= 0:
        return params_list[0]

    # Calculate pairwise Euclidean distances between parameter sets
    distances = torch.zeros((num_neighbors, num_neighbors), device=device)
    for i in range(num_neighbors):
        for j in range(i + 1, num_neighbors):
            # Calculate distance between parameter sets
            dist = 0
            for param_idx in range(len(params_list[0])):
                param_i = params_list[i][param_idx].reshape(-1)
                param_j = params_list[j][param_idx].reshape(-1)
                dist += torch.sum((param_i - param_j) ** 2)

            # Store the square root of the distance
            distances[i, j] = distances[j, i] = torch.sqrt(dist)

    # Calculate scores for each parameter set
    scores = torch.zeros(num_neighbors, device=device)
    for i in range(num_neighbors):
        # Find indices of closest neighbors
        closest_indices = torch.argsort(distances[i])[:num_to_select + 1]
        # Sum distances to closest neighbors
        scores[i] = torch.sum(distances[i, closest_indices])

    # Select parameter set with minimum score
    selected_index = torch.argmin(scores).item()
    return params_list[selected_index]


def krum_trimmed_mean_screen(params_list, trim_param, num_byzantine):
    """BRIDGE-B: Krum followed by Trimmed Mean"""
    if len(params_list) <= trim_param * 2 + 1:
        # Not enough parameters for proper trimming, use Krum only
        return krum_screen(params_list, num_byzantine)

    # First, select a parameter set using Krum
    krum_selected_params = krum_screen(params_list, num_byzantine)

    # Find index of the selected parameter set
    selected_index = -1
    for i, params in enumerate(params_list):
        if all(torch.allclose(krum_selected_params[p], params[p]) for p in range(len(params))):
            selected_index = i
            break

    # If we couldn't find the index, use all parameters
    if selected_index == -1:
        trimmed_mean_params_list = params_list
    else:
        # Remove the selected parameters and perform trimmed mean on the rest
        trimmed_mean_params_list = [params for i, params in enumerate(params_list) if i != selected_index]

    return trimmed_mean_screen(trimmed_mean_params_list, trim_param)


# --- Byzantine attack strategies ---
def random_attack(params, device):
    """Random values attack"""
    return [torch.randn_like(p).to(device) for p in params]


def sign_flipping_attack(params, device):
    """Sign flipping attack - flips the sign of gradients"""
    return [-1.0 * p for p in params]


def get_byzantine_params(original_params, attack_type, device):
    if attack_type == "random":
        return random_attack(original_params, device)
    elif attack_type == "sign_flipping":
        return sign_flipping_attack(original_params, device)
    else:
        return original_params  # Default case


# --- Training and evaluation functions ---
def train_epoch(models, trainloaders, adj_matrix, byzantine_indices, criterion, current_lr, variants,
                attack_type="random"):
    epoch_losses = {variant: [[] for _ in range(num_nodes)] for variant in variants}

    # Determine maximum number of batches
    max_batches = min([len(loader) for loader in trainloaders])

    for batch_idx in range(max_batches):
        # 1. Each node computes local gradients
        local_gradients = {variant: [] for variant in variants}
        for node_idx in range(num_nodes):
            # Get batch data
            data_iter = iter(trainloaders[node_idx])
            try:
                data, target = next(data_iter)
            except StopIteration:
                # Reset data iterator if it's exhausted
                data_iter = iter(trainloaders[node_idx])
                data, target = next(data_iter)

            data, target = data.to(device), target.to(device)

            for variant in variants:
                model = models[variant][node_idx]
                model.train()

                # Forward pass
                model.zero_grad()
                output = model(data)
                loss = criterion(output, target)

                # Backward pass
                loss.backward()

                # Record loss
                epoch_losses[variant][node_idx].append(loss.item())

                # Collect gradients
                grads = [param.grad.clone() for param in model.parameters()]
                local_gradients[variant].append(grads)

        # 2. Broadcast model parameters
        all_params = {variant: [] for variant in variants}
        for node_idx in range(num_nodes):
            for variant in variants:
                model_params = [param.data.clone() for param in models[variant][node_idx].parameters()]

                # Apply Byzantine attack if this is a Byzantine node
                if node_idx in byzantine_indices:
                    model_params = get_byzantine_params(model_params, attack_type, device)

                all_params[variant].append(model_params)

        # 3. Receive and filter parameters
        filtered_params = {variant: [] for variant in variants}
        for node_idx in range(num_nodes):
            # Get indices of neighbors (including self)
            neighbor_indices = np.where(adj_matrix[node_idx])[0]

            for variant in variants:
                # Get parameters from neighbors
                neighbor_params = [all_params[variant][i] for i in neighbor_indices]

                # Apply screening functions based on variant
                if variant == "BRIDGE-T":
                    aggregated_params = trimmed_mean_screen(neighbor_params, trim_parameter)
                elif variant == "BRIDGE-M":
                    aggregated_params = median_screen(neighbor_params)
                elif variant == "BRIDGE-K":
                    aggregated_params = krum_screen(neighbor_params, len(byzantine_indices))
                elif variant == "BRIDGE-B":
                    aggregated_params = krum_trimmed_mean_screen(neighbor_params, trim_parameter,
                                                                 len(byzantine_indices))
                else:
                    raise ValueError(f"Unknown variant: {variant}")

                filtered_params[variant].append(aggregated_params)

        # 4. Update models
        for node_idx in range(num_nodes):
            # Skip Byzantine nodes (their parameters are irrelevant for evaluation)
            if node_idx not in byzantine_indices:
                for variant in variants:
                    # Manual update using filtered parameters and local gradients
                    for param, agg_param, grad in zip(models[variant][node_idx].parameters(),
                                                      filtered_params[variant][node_idx],
                                                      local_gradients[variant][node_idx]):
                        # Update: param = aggregated_param - lr * gradient
                        param.data = agg_param - current_lr * grad

    # Calculate mean loss for each node and variant
    mean_losses = {variant: [np.mean(losses) if losses else float('inf')
                             for losses in epoch_losses[variant]]
                   for variant in variants}

    return mean_losses


def evaluate_models(models, testloader, byzantine_indices, variants):
    accuracies = {variant: [] for variant in variants}

    for variant in variants:
        for node_idx in range(num_nodes):
            # Skip Byzantine nodes
            if node_idx not in byzantine_indices:
                model = models[variant][node_idx]
                model.eval()

                correct = 0
                total = 0

                with torch.no_grad():
                    for data, target in testloader:
                        data, target = data.to(device), target.to(device)

                        # Forward pass
                        outputs = model(data)
                        _, predicted = torch.max(outputs.data, 1)

                        # Calculate accuracy
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

                accuracy = 100 * correct / total
                accuracies[variant].append(accuracy)

    return accuracies


# --- Visualization functions ---
def plot_results(all_epoch_losses, all_epoch_accuracies, adj_matrix, graph, byzantine_indices, variants):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = f"results_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    # Create figure for accuracy and loss plots
    plt.figure(figsize=(18, 12))

    # Plot accuracy curves for each variant
    for i, variant in enumerate(variants):
        plt.subplot(2, len(variants), i + 1)  # Adjust layout based on number of variants
        
        # Plot accuracy for each non-byzantine node
        for node_idx in range(num_nodes):
            if node_idx not in byzantine_indices:
                node_accuracies = all_epoch_accuracies[variant][node_idx]
                if node_accuracies:  # Check if there are accuracies recorded
                    epochs = list(range(1, len(node_accuracies) + 1))
                    plt.plot(epochs, node_accuracies, label=f"Node {node_idx}")

        plt.xlabel("Evaluation Point")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy ({variant})")
        plt.legend()
        plt.grid(True)

    # Plot loss curves for each variant
    for i, variant in enumerate(variants):
        plt.subplot(2, len(variants), i + len(variants) + 1)  # Position in bottom row
        
        # Plot loss for each non-byzantine node
        for node_idx in range(num_nodes):
            if node_idx not in byzantine_indices:
                node_losses = all_epoch_losses[variant][node_idx]
                if node_losses:  # Check if there are losses recorded
                    epochs = list(range(1, len(node_losses) + 1))
                    plt.plot(epochs, node_losses, label=f"Node {node_idx}")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss ({variant})")
        plt.yscale("log")  # Use logarithmic scale for loss
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/accuracy_loss_comparison.png", dpi=300)

    # Create a separate figure for network topology
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph, seed=seed)

    # Draw normal nodes
    non_byz_nodes = [i for i in range(num_nodes) if i not in byzantine_indices]
    nx.draw_networkx_nodes(graph, pos, nodelist=non_byz_nodes, node_color='blue', node_size=300, alpha=0.8)

    # Draw Byzantine nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=byzantine_indices, node_color='red', node_size=300, alpha=0.8)

    # Draw network connections
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')

    plt.title("Network Topology (Red: Byzantine, Blue: Honest)")
    plt.axis('off')
    plt.savefig(f"{result_dir}/network_topology.png", dpi=300)

    # Show all figures
    plt.show()

    print(f"Results saved to {result_dir}")
    
# --- Main function ---
def main():
    print("Starting Byzantine-resilient federated learning experiment...")

    # Set up variants
    variants = ["BRIDGE-T", "BRIDGE-M", "BRIDGE-K", "BRIDGE-B"]

    # Load data
    trainloaders, testloader = load_data()
    print("Data loaded successfully")

    # Create network topology
    adj_matrix, graph = create_adjacency_matrix(num_nodes, connectivity, seed)
    print("Network topology created")
    print(f"Adjacency Matrix:\n{adj_matrix}")

    # Select Byzantine nodes
    byzantine_indices = select_byzantine_nodes(num_nodes, max_byzantine_nodes)

    # Initialize models for each variant and node
    models = {variant: [SimpleCNN().to(device) for _ in range(num_nodes)] for variant in variants}
    print("Models initialized")

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize storage for losses and accuracies
    all_epoch_losses = {variant: [[] for _ in range(num_nodes)] for variant in variants}
    all_epoch_accuracies = {variant: [[] for _ in range(num_nodes)] for variant in variants}

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        # Get current learning rate
        current_lr = lr_schedule(epoch)

        # Train one epoch
        mean_losses = train_epoch(models, trainloaders, adj_matrix, byzantine_indices,
                                  criterion, current_lr, variants)

        # Record losses
        for variant in variants:
            for node_idx in range(num_nodes):
                all_epoch_losses[variant][node_idx].append(mean_losses[variant][node_idx])

        # Evaluate models every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            accuracies = evaluate_models(models, testloader, byzantine_indices, variants)

            # Record accuracies
            for variant in variants:
                for node_idx in range(num_nodes):
                    if node_idx not in byzantine_indices:
                        all_epoch_accuracies[variant][node_idx].append(
                            accuracies[variant][node_idx - len(byzantine_indices)])

            # Print current progress
            for variant in variants:
                honest_accuracies = [acc for i, acc in enumerate(accuracies[variant])]
                mean_acc = np.mean(honest_accuracies)
                print(f"  {variant}: Mean accuracy = {mean_acc:.2f}%")

    print("Training completed")

    # Plot and save results
    plot_results(all_epoch_losses, all_epoch_accuracies, adj_matrix, graph, byzantine_indices, variants)


if __name__ == "__main__":
    main()