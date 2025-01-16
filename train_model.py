# train_model.py
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Load model
model = resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 10)  # Update for 10 CIFAR-10 classes
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):  # Few epochs for demonstration
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), 'cifar10_model.pth')
