import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

EPOCHS = 10
BATCH_SIZE = 128
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


def train_model(model, train_loader, val_loader, optimizer, criterion):
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    for epoch in range(EPOCHS):
        # Training
        model.train()
        correct, total, running_loss = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        train_loss_list.append(running_loss / len(train_loader))
        train_acc_list.append(correct / total)

        # Validation
        model.eval()
        correct, total, running_loss = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                running_loss += loss.item()
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        val_loss_list.append(running_loss / len(val_loader))
        val_acc_list.append(correct / total)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Acc: {train_acc_list[-1]:.4f} Val Acc: {val_acc_list[-1]:.4f} | "
              f"Train Loss: {train_loss_list[-1]:.4f} Val Loss: {val_loss_list[-1]:.4f}")

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list


model = CNN().to(device)
learning_rate = 0.001  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, test_loader, optimizer, criterion)


best_epoch = val_acc.index(max(val_acc)) + 1
print(f"\nBest Epoch: {best_epoch} | Val Acc: {val_acc[best_epoch-1]:.4f}")


plt.figure()
plt.plot(range(1, EPOCHS+1), train_loss, label="Train Loss")
plt.plot(range(1, EPOCHS+1), val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MNIST CNN - Loss per Epoch")
plt.legend()
plt.savefig("results/MNIST_loss.png")
plt.close()

plt.figure()
plt.plot(range(1, EPOCHS+1), train_acc, label="Train Acc")
plt.plot(range(1, EPOCHS+1), val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST CNN - Accuracy per Epoch")
plt.legend()
plt.savefig("results/MNIST_acc.png")
plt.close()
