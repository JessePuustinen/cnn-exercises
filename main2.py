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
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, optimizer, criterion, tag):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

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
        train_loss.append(running_loss / len(train_loader))
        train_acc.append(correct / total)

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
        val_loss.append(running_loss / len(val_loader))
        val_acc.append(correct / total)

        print(f"{tag} | Epoch {epoch+1}/{EPOCHS} "
              f"Train Acc: {train_acc[-1]:.4f} Val Acc: {val_acc[-1]:.4f}")

    return train_loss, val_loss, train_acc, val_acc


def plot_results(results, ylabel, title, filename):
    plt.figure()
    for label, values in results.items():
        plt.plot(values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


sgd_params = [
    {"lr":0.01, "momentum":0.0},
    {"lr":0.01, "momentum":0.9},
    {"lr":0.1, "momentum":0.0},
    {"lr":0.1, "momentum":0.9},
]

sgd_results = {}
criterion = nn.CrossEntropyLoss()
best_val_acc = 0
best_sgd_label = None
best_sgd_results = None

for p in sgd_params:
    label = f"SGD_lr{p['lr']}_mom{p['momentum']}"
    print(f"\nRunning {label}")
    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=p['lr'], momentum=p['momentum'])
    tl, vl, ta, va = train_model(model, train_loader, test_loader, optimizer, criterion, label)
    sgd_results[label+"_loss"] = vl
    sgd_results[label+"_acc"] = va

    # track best
    max_val = max(va)
    if max_val > best_val_acc:
        best_val_acc = max_val
        best_sgd_label = label
        best_sgd_results = (vl, va)

# Save SGD results
with open("results/sgd_results.pkl", "wb") as f:
    pickle.dump(sgd_results, f)

# Plot the best SGD
plt.figure()
vl, va = best_sgd_results
plt.plot(vl, label="Validation Loss")
plt.plot(va, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title(f"Best SGD Hyperparameters: {best_sgd_label}")
plt.legend()
plt.savefig("results/SGD_best.png")
plt.close()
print(f"\nBest SGD: {best_sgd_label} | Val Acc: {best_val_acc:.4f}")


optimizer_params = {
    "RMSProp": [
        {"lr":0.001, "momentum":0.0},
        {"lr":0.001, "momentum":0.9},
        {"lr":0.01,  "momentum":0.9},
    ],
    "AdaGrad": [
        {"lr":0.001},
        {"lr":0.01},
        {"lr":0.1},
    ],
    "Adam": [
        {"lr":0.0005},
        {"lr":0.001},
        {"lr":0.005},
    ]
}

opt_results = {}
best_opt_results = {}

for name, params_list in optimizer_params.items():
    best_val = 0
    best_label = ""
    best_res = None
    for p in params_list:
        param_str = "_".join([f"{k}{v}" for k,v in p.items()])
        label = f"{name}_{param_str}"
        print(f"\nRunning {label}")
        model = CNN().to(device)
        if name == "RMSProp":
            optimizer = optim.RMSprop(model.parameters(), lr=p['lr'], momentum=p['momentum'])
        elif name == "AdaGrad":
            optimizer = optim.Adagrad(model.parameters(), lr=p['lr'])
        elif name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=p['lr'])
        else:
            raise ValueError("Unknown optimizer")
        tl, vl, ta, va = train_model(model, train_loader, test_loader, optimizer, criterion, label)
        opt_results[label+"_loss"] = vl
        opt_results[label+"_acc"] = va

        # track best for this optimizer
        max_val = max(va)
        if max_val > best_val:
            best_val = max_val
            best_label = label
            best_res = (vl, va)

    best_opt_results[name] = (best_label, best_res, best_val)
    print(f"\nBest {name}: {best_label} | Val Acc: {best_val:.4f}")

# Save optimizer results
with open("results/opt_results.pkl", "wb") as f:
    pickle.dump(opt_results, f)


# SGD tuning plot
plot_results(sgd_results, "Validation Accuracy", "SGD Hyperparameter Tuning", "results/SGD_tuning_acc.png")
plot_results(sgd_results, "Validation Loss", "SGD Hyperparameter Tuning", "results/SGD_tuning_loss.png")

# Optimizer comparison 
best_labels = {name: best_opt_results[name][0] for name in best_opt_results}
best_values_acc = {name: best_opt_results[name][1][1] for name in best_opt_results}  # val_acc
best_values_loss = {name: best_opt_results[name][1][0] for name in best_opt_results} # val_loss

plot_results(best_values_acc, "Validation Accuracy", "Optimizer Comparison (Best Params)", "results/Optimizer_comparison_acc.png")
plot_results(best_values_loss, "Validation Loss", "Optimizer Comparison (Best Params)", "results/Optimizer_comparison_loss.png")
