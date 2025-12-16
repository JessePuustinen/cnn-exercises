import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        
        if input_channels == 3:  # CIFAR10 32x32
            self.fc1 = nn.Linear(64*8*8, 128)
        else:  # MNIST 28x28
            self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    train_acc_list, val_acc_list = [], []
    for epoch in range(epochs):
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
        train_acc = correct / total
        train_acc_list.append(train_acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = torch.max(out,1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        val_acc = correct / total
        val_acc_list.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    return train_acc_list, val_acc_list


cifar_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                              ]))
cifar_train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True)
cifar_test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False)

cifar_model = CNN(input_channels=3, num_classes=10).to(device)
optimizer = optim.Adam(cifar_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_model(cifar_model, cifar_train_loader, cifar_test_loader, optimizer, criterion, epochs=10)

# Test dog.jpg
dog_img = Image.open("dog.jpg").convert('RGB')
dog_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
dog_tensor = dog_transform(dog_img).unsqueeze(0).to(device)
cifar_model.eval()
with torch.no_grad():
    output = cifar_model(dog_tensor)
    _, pred = torch.max(output,1)
print("Predicted CIFAR-10 class for dog.jpg:", cifar_train.classes[pred.item()])


mnist_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

mnist_train = datasets.MNIST('./data', train=True, download=True, transform=mnist_transform)
mnist_test = datasets.MNIST('./data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,),(0.5,))
                             ]))
mnist_train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

mnist_model = CNN(input_channels=1, num_classes=10).to(device)
optimizer = optim.Adam(mnist_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_model(mnist_model, mnist_train_loader, mnist_test_loader, optimizer, criterion, epochs=10)

# Test handwritten6.jpg
hw_img = Image.open("handwritten6.jpg").convert('L')
hw_transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
hw_tensor = hw_transform(hw_img).unsqueeze(0).to(device)
mnist_model.eval()
with torch.no_grad():
    output = mnist_model(hw_tensor)
    _, pred = torch.max(output,1)
print("Predicted MNIST class for handwritten6.jpg:", pred.item())
