import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
import matplotlib.pyplot as plt
import random

# ==================================
# 1. USTAWIENIA
# ==================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Używane urządzenie:", device)

batch_size = 32
epochs = 5
learning_rate = 0.0003

# ==================================
# 2. AUGMENTACJA DANYCH
# ==================================

transform_train = transforms.Compose([

    # mniejszy resize -> dużo szybciej na CPU
    transforms.Resize((64, 64)),

    # augmentacja
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

transform_test = transforms.Compose([

    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

# ==================================
# 3. CIFAR-10
# ==================================

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
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

classes = [
    "samolot",
    "samochód",
    "ptak",
    "kot",
    "jeleń",
    "pies",
    "żaba",
    "koń",
    "statek",
    "ciężarówka"
]

# ==================================
# 4. PRETRAINED RESNET18
# ==================================

model = timm.create_model(
    'resnet18',
    pretrained=True,
    num_classes=10
)

model = model.to(device)

print(model)

# ==================================
# 5. LOSS + OPTIMIZER
# ==================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

# ==================================
# 6. TRENING
# ==================================

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

    scheduler.step()

    avg_loss = running_loss / len(train_loader)

    train_losses.append(avg_loss)

    # ==================================
    # TEST
    # ==================================

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

    print(
        f"Epoka [{epoch+1}/{epochs}] "
        f"| Loss: {avg_loss:.4f} "
        f"| Accuracy: {accuracy:.2f}%"
    )

# ==================================
# 7. WYKRESY
# ==================================

plt.figure(figsize=(8, 5))

plt.plot(train_losses)

plt.title("Strata podczas treningu")

plt.xlabel("Epoka")

plt.ylabel("Loss")

plt.grid()

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(test_accuracies)

plt.title("Accuracy na CIFAR-10")

plt.xlabel("Epoka")

plt.ylabel("Accuracy [%]")

plt.grid()

plt.show()

# ==================================
# 8. TEST LOSOWEGO OBRAZU
# ==================================

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

plt.figure(figsize=(4, 4))

plt.imshow(image_to_show)

plt.title(
    f"Prawdziwa klasa: {classes[label]}\n"
    f"Predykcja: {classes[predicted.item()]}"
)

plt.axis("off")

plt.show()

# ==================================
# 9. TEST NA SVHN
# ==================================

print("\nŁadowanie SVHN...")

svhn_dataset = torchvision.datasets.SVHN(
    root="./data",
    split='test',
    download=True,
    transform=transform_test
)

svhn_loader = torch.utils.data.DataLoader(
    svhn_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in svhn_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

svhn_accuracy = 100 * correct / total

print(f"\nAccuracy na SVHN: {svhn_accuracy:.2f}%")
