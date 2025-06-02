import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_train_data(X,y,test_size,batch_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)  # (N,) → (N,1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader,test_loader


def train_model(model,epoch,train_loader):
    losses=[]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epoch):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            # print()
            preds = model(xb)
            # print(preds.reshape(-1,1))
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(loss)
        losses.append(epoch_loss)
    return model,losses

def plot_loss_accuracy(losses, epochs):
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))

    # Plot Loss
    axes.plot(range(1, epochs + 1), losses, marker='o', linestyle='-', color='tab:red', linewidth=2)
    axes.set_title("📉 Training Loss vs Epochs", fontsize=16)
    axes.set_xlabel("Epoch", fontsize=14)
    axes.set_ylabel("Loss", fontsize=14)
    axes.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    return fig