from torchvision import datasets,transforms
import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn


data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)



##transform the images in dataset to tensors of torch

##precomputed mean
mean = [0.4914, 0.4822, 0.4465]  # for CIFAR-10 RGB channels

std  = [0.2023, 0.1994, 0.2010]


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Re-load with this transform
cifar10 = torchvision.datasets.CIFAR10(root="...", train=True, transform=transform, download=True)


##flatten all the tensors of imgs to feed to the neural network


train_loader = DataLoader(cifar10, batch_size=64, shuffle=True)



model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)



criterion =torch. nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# --- 6. Training Loop ---
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


##Testing the model


img,label=cifar2[0]

img_tensor = transform(img).unsqueeze(0)

model.eval()

with torch.no_grad():
  output=model(img_tensor)
  predicted_class=torch.argmax(output,dim=1).item()


torch.save(model, "cifar_full_model.pth")


