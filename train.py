import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_cifar10_loaders
from model import SimpleCNN
from utils import save_model

def train_model(epochs=10, batch_size=64, lr=0.001):
    trainloader, _ = get_cifar10_loaders(batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f}")

    save_model(model)

if __name__ == '__main__':
    train_model()
