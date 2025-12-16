import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


EPOCHS = 5
BATCH_SIZE = 128
LR = 0.001

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)


class CNN(nn.Module):
    def __init__(self, activation, in_channels):
        super().__init__()
        self.act = activation
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        if in_channels == 1:      # MNIST
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
        else:                     # CIFAR-10
            self.fc1 = nn.Linear(64 * 8 * 8, 128)

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


def train_model(model, train_loader, val_loader, optimizer, criterion, tag):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(EPOCHS):
        
        model.train()
        correct, total, running_loss = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(correct / total)

        
        model.eval()
        correct, total, running_loss = 0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += y.size(0)
                correct += (preds == y).sum().item()

        val_loss.append(running_loss / len(val_loader))
        val_acc.append(correct / total)

        print(f"{tag} | Epoch {epoch+1}/{EPOCHS} "
              f"Train Acc: {train_acc[-1]:.4f} "
              f"Val Acc: {val_acc[-1]:.4f}")

    return train_loss, val_loss, train_acc, val_acc


def plot_all_curves(results, ylabel, title, filename):
    plt.figure()
    for name, values in results.items():
        plt.plot(values, label=name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


activations = {
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "ReLU": nn.ReLU(),
    "ELU": nn.ELU(),
    "SELU": nn.SELU()
}


def run_experiment(dataset_name):
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
        in_channels = 1
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        in_channels = 3

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    all_train_loss = {}
    all_val_loss = {}
    all_train_acc = {}
    all_val_acc = {}

    for name, act in activations.items():
        model_file = f"results/{dataset_name}_{name}_model.pt"
        metric_file = f"results/{dataset_name}_{name}_metrics.pkl"

        if os.path.exists(metric_file):
            print(f"{dataset_name} | {name}: Loading saved results")
            with open(metric_file, "rb") as f:
                tl, vl, ta, va = pickle.load(f)
        else:
            print(f"{dataset_name} | {name}: Training")
            model = CNN(act, in_channels).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()

            tl, vl, ta, va = train_model(
                model, train_loader, test_loader,
                optimizer, criterion,
                f"{dataset_name}-{name}"
            )

            torch.save(model.state_dict(), model_file)
            with open(metric_file, "wb") as f:
                pickle.dump((tl, vl, ta, va), f)

        all_train_loss[name] = tl
        all_val_loss[name] = vl
        all_train_acc[name] = ta
        all_val_acc[name] = va

    
    plot_all_curves(all_train_loss, "Loss",
                    f"{dataset_name} Training Loss",
                    f"results/{dataset_name}_train_loss.png")

    plot_all_curves(all_val_loss, "Loss",
                    f"{dataset_name} Validation Loss",
                    f"results/{dataset_name}_val_loss.png")

    plot_all_curves(all_train_acc, "Accuracy",
                    f"{dataset_name} Training Accuracy",
                    f"results/{dataset_name}_train_acc.png")

    plot_all_curves(all_val_acc, "Accuracy",
                    f"{dataset_name} Validation Accuracy",
                    f"results/{dataset_name}_val_acc.png")


if __name__ == "__main__":
    run_experiment("MNIST")
    run_experiment("CIFAR10")
