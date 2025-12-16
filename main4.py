import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.act(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    train_errors, val_errors = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        train_error = 100 * (1 - correct / total)
        train_errors.append(train_error)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        val_error = 100 * (1 - correct / total)
        val_errors.append(val_error)

        print(f"Epoch {epoch+1}/{epochs} | Train Error: {train_error:.2f}% | Val Error: {val_error:.2f}%")

    return train_errors, val_errors


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)


# Model without BN
model_base = CNN().to(device)
optimizer = optim.Adam(model_base.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_error_base, val_error_base = train_model(model_base, train_loader, test_loader, optimizer, criterion)

# Model with BN
model_bn = CNN_BN().to(device)
optimizer = optim.Adam(model_bn.parameters(), lr=0.001)
train_error_bn, val_error_bn = train_model(model_bn, train_loader, test_loader, optimizer, criterion)


os.makedirs("results", exist_ok=True)

plt.figure()
plt.plot(train_error_base, label='Train w/o BN')
plt.plot(val_error_base, label='Val w/o BN')
plt.plot(train_error_bn, label='Train w/ BN')
plt.plot(val_error_bn, label='Val w/ BN')
plt.xlabel('Epoch')
plt.ylabel('Classification Error (%)')
plt.title('CNN with vs without Batch Normalization')
plt.legend()
plt.savefig('results/MNIST_BN_comparison.png')
plt.show()
