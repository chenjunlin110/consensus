import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
import networkx as nx
import random
import math
import torch.nn.functional as F

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

num_agents = 10
epochs = 100
learning_rate = 0.01  # Basic learning rate for gradient descent
batch_size = 64
connectivity = 0.5
byzantine_agents = random.sample(range(num_agents), int(num_agents * 0.2))  # 选取 20% 代理作为拜占庭节点

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Split data among agents (simple splitting for stability)
indices = list(range(len(dataset)))
random.shuffle(indices)
split_indices = np.array_split(indices, num_agents)
subsets = [Subset(dataset, idx) for idx in split_indices]
dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Simplified architecture for stability
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def initialize_weights(self):
        # Use more conservative initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)  # Smaller gain for stability
                nn.init.constant_(m.bias, 0)
def create_weighted_doubly_stochastic_matrix(num_agents, prob):
    """
    1. Create an Erdős-Rényi random graph with a given connection probability.
    2. Assign random positive weights to edges.
    3. Transform the weight matrix into a doubly stochastic matrix using Sinkhorn-Knopp.
    """
    # Step 1: Generate Erdős-Rényi adjacency matrix
    G = nx.erdos_renyi_graph(num_agents, prob)
    A = nx.to_numpy_array(G)  # Convert to adjacency matrix
    
    # Step 2: Assign random positive weights where there is an edge
    W = np.random.rand(num_agents, num_agents) * A  # Assign weights only to existing edges
    np.fill_diagonal(W, 1)  # Ensure self-loops have weights
    
    # Step 3: Apply Sinkhorn-Knopp to make W doubly stochastic
    max_iter = 1000
    tol = 1e-8
    
    for _ in range(max_iter):
        row_sums = W.sum(axis=1, keepdims=True)
        W /= row_sums  # Normalize rows
        col_sums = W.sum(axis=0, keepdims=True)
        W /= col_sums  # Normalize columns
        
        # Check convergence
        if np.allclose(W.sum(axis=1), 1, atol=tol) and np.allclose(W.sum(axis=0), 1, atol=tol):
            break
    
    print(W)
    # Convert to PyTorch tensor
    return torch.tensor(W, dtype=torch.float32)



def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def run_decentralized_learning(epochs, num_agents, dataloaders, val_loader, device, lr=0.01):
    """Run decentralized learning with basic gradient descent"""
    # Initialize models
    

    models = [CNN().to(device) for _ in range(num_agents)]
    for model in models:
        model.initialize_weights()
    
    # Loss function - Cross Entropy for all models
    criterions = [nn.CrossEntropyLoss() for _ in range(num_agents)]
    
    # Initialize metrics tracking
    train_losses = [[] for _ in range(num_agents)]
    val_accuracies = [[] for _ in range(num_agents)]
    consensus_metrics = []
    
    # Initialize doubly stochastic matrix for mixing
    W = create_weighted_doubly_stochastic_matrix(num_agents, connectivity).to(device)
    
    # Training loop
    for epoch in range(epochs):
        # Local training step
        for i in range(num_agents):
            models[i].train()
            running_loss = 0.0
            
            for images, labels in dataloaders[i]:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = models[i](images)
                loss = criterions[i](outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Manual gradient descent update
                with torch.no_grad():
                    for param in models[i].parameters():
                        if param.grad is not None:
                            param.data.sub_(lr * param.grad)
                
                # Zero gradients manually
                for param in models[i].parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                
                # Check for NaN values in model parameters
                if any(torch.isnan(param).any() for param in models[i].parameters()):
                    print(f"NaN detected in Agent {i+1} parameters after backward step")
                    # Reinitialize the model if needed
                    models[i] = CNN().to(device)
                    models[i].initialize_weights()
                    break
                
                running_loss += loss.item()
            
            # Store loss safely
            epoch_loss = running_loss / len(dataloaders[i])
            if not np.isnan(epoch_loss) and not np.isinf(epoch_loss):
                train_losses[i].append(epoch_loss)
            else:
                if len(train_losses[i]) > 0:
                    train_losses[i].append(train_losses[i][-1])  # Use previous loss
                else:
                    train_losses[i].append(10.0)  # Default value
                print(f"Warning: Agent {i+1} has invalid loss value at epoch {epoch+1}")
        
        # Consensus step with doubly stochastic matrix
        with torch.no_grad():
            # Safe parameter averaging
            param_dict = {i: {} for i in range(num_agents)}
            
            # First collect all parameters
            for i, model in enumerate(models):
                for name, param in model.named_parameters():
                    param_dict[i][name] = param.data.clone()
            
            # Perform weighted average for each parameter
            for i, model in enumerate(models):
                for name, param in model.named_parameters():
                    aggregated_param = torch.zeros_like(param.data)
                    
                    for j in range(num_agents):
                       if W[i, j] > 0:
                        aggregated_param += W[i, j] * param_dict[j][name]

                    param.data.copy_(aggregated_param)
        
        # Evaluate and log (every 10 epochs)
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Calculate consensus metric safely
            param_variance = 0.0
            
            # Get flattened parameters
            flattened_params = []
            for model in models:
                agent_params = []
                for name, param in model.named_parameters():
                    if not torch.isnan(param).any():
                        agent_params.append(param.data.flatten())
                
                if agent_params:
                    flattened_params.append(torch.cat(agent_params))
            
            # Calculate variance if we have valid parameters
            if flattened_params:
                stacked_params = torch.stack(flattened_params)
                param_variance = torch.var(stacked_params, dim=0).mean().item()
                
            consensus_metrics.append(param_variance)
            print(f"Epoch {epoch+1}, Consensus metric: {param_variance:.6f}")
            
            # Evaluate all agents
            for i, model in enumerate(models):
                if not any(torch.isnan(param).any() for param in model.parameters()):
                    accuracy = evaluate_model(model, val_loader, device)
                    val_accuracies[i].append(accuracy)
                    print(f"Agent {i+1}, Epoch {epoch+1}, Loss: {train_losses[i][-1]:.4f}, Val Acc: {accuracy:.2f}%")
                else:
                    if val_accuracies[i]:
                        val_accuracies[i].append(val_accuracies[i][-1])
                    else:
                        val_accuracies[i].append(0.0)
                    print(f"Agent {i+1}, Epoch {epoch+1}, Loss: {train_losses[i][-1]:.4f}, Val Acc: N/A (NaN params)")
    
    return train_losses, val_accuracies, consensus_metrics, W

# Main execution
if __name__ == '__main__':
    print(f"Running decentralized learning with {num_agents} agents for {epochs} epochs")
    
    # Run with basic gradient descent
    train_losses, val_accuracies, consensus_metrics, mixing_matrix = run_decentralized_learning(
        epochs=epochs,
        num_agents=num_agents,
        dataloaders=dataloaders,
        val_loader=val_loader,
        device=device,
        lr=0.01  # Basic gradient descent learning rate
    )
    
    # Plot results
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Training loss (safe plotting)
    plt.subplot(2, 3, 1)
    for i in range(num_agents):
        # Filter out extreme values
        safe_losses = np.array(train_losses[i])
        safe_losses = np.clip(safe_losses, 0, 10)  # Clip to reasonable range
        plt.plot(safe_losses, label=f"Agent {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Validation accuracy
    plt.subplot(2, 3, 2)
    eval_points = list(range(0, epochs, 10)) + ([epochs-1] if (epochs-1) % 10 != 0 else [])
    for i in range(num_agents):
        plt.plot(eval_points, val_accuracies[i], marker='o', label=f"Agent {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Consensus metric
    plt.subplot(2, 3, 3)
    plt.plot(eval_points, consensus_metrics, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Parameter Variance")
    plt.title("Consensus Metric")
    plt.grid(True)
    
    # Plot 4: Final model comparison
    plt.subplot(2, 3, 4)
    final_accuracies = [val_accuracies[i][-1] for i in range(num_agents)]
    plt.bar(range(1, num_agents+1), final_accuracies)
    plt.xlabel("Agent")
    plt.ylabel("Final Accuracy (%)")
    plt.title("Final Model Performance")
    plt.xticks(range(1, num_agents+1))
    plt.grid(True, axis='y')
    
    # Plot 5: Mixing Matrix Visualization
    plt.subplot(2, 3, 5)
    W_np = mixing_matrix.cpu().numpy()
    plt.imshow(W_np, cmap='viridis')
    plt.colorbar(label='Weight')
    plt.title("Doubly Stochastic Mixing Matrix")
    plt.xlabel("Agent j")
    plt.ylabel("Agent i")
    
    plt.tight_layout()
    plt.savefig("decentralized_learning_results.png")
    plt.show()












