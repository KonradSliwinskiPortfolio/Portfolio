import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

photo_size = (256, 256)

model = models.resnet50(pretrained=False)  

number_class = 5  
model.fc = nn.Linear(model.fc.in_features, number_class)

transform = transforms.Compose([
    transforms.Resize(photo_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dane = datasets.ImageFolder(root="RESNET/glowice2", transform=transform)
train_loader = torch.utils.data.DataLoader(train_dane, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 50
for epoch in range(num_epochs):
    print("epoka: ",epoch)
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "trained_model50_502.pth")