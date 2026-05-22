import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random


# =========================
# 1. USTAWIENIA
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Używane urządzenie:", device)

batch_size = 128
epochs = 15
learning_rate = 0.00005


# =========================
# 2. DANE CIFAR-10
# =========================

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

classes = [
    "samolot", "samochód", "ptak", "kot", "jeleń",
    "pies", "żaba", "koń", "statek", "ciężarówka"
]
# =========================
# 3. WŁASNA SIEĆ VGG11
# =========================

class MyVGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(MyVGG11, self).__init__()

        self.features = nn.Sequential(
            # blok 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            # blok 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8

            # blok 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4

            # blok 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4 -> 2x2

            # blok 5
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 2x2 -> 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = MyVGG11(num_classes=10).to(device)
print(model)
# =========================
# 4. FUNKCJA STRATY I OPTYMALIZATOR
# =========================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# =========================
# 5. TRENING MODELU
# =========================

train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # test po każdej epoce
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoka [{epoch+1}/{epochs}] | Strata: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    
    # =========================
# 6. WYKRES STRATY I ACCURACY
# =========================

plt.figure()
plt.plot(train_losses)
plt.title("Strata podczas treningu")
plt.xlabel("Epoka")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(test_accuracies)
plt.title("Dokładność na zbiorze testowym")
plt.xlabel("Epoka")
plt.ylabel("Accuracy [%]")
plt.show()
# =========================
# 7. TEST NA LOSOWYM OBRAZKU
# =========================

def unnormalize(img):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return img.cpu() * std + mean


model.eval()

random_index = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[random_index]

with torch.no_grad():
    input_image = image.unsqueeze(0).to(device)
    output = model(input_image)
    _, predicted = torch.max(output, 1)

image_to_show = unnormalize(image)
image_to_show = image_to_show.permute(1, 2, 0)
image_to_show = torch.clamp(image_to_show, 0, 1)

plt.imshow(image_to_show)
plt.title(
    f"Prawdziwa klasa: {classes[label]}\n"
    f"Predykcja sieci: {classes[predicted.item()]}"
)
plt.axis("off")
plt.show()
