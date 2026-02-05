import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import time
import copy

DATA_DIR = './dataset'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
}

def load_data(data_dir):
    full_dataset = datasets.ImageFolder(
        data_dir,
        transform=data_transforms['train']
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    }

    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    return dataloaders, class_names, len(full_dataset)

def initialize_model(num_classes):
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model.to(DEVICE)

def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=10
):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(
                    preds == labels.data
                )

            epoch_loss = (
                running_loss /
                len(dataloaders[phase].dataset)
            )
            epoch_acc = (
                running_corrects.double() /
                len(dataloaders[phase].dataset)
            )

            print(
                f'{phase} Loss: {epoch_loss:.4f} '
                f'Acc: {epoch_acc:.4f}'
            )

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(
                    model.state_dict()
                )

            if phase == 'val':
                val_acc_history.append(
                    epoch_acc.cpu().numpy()
                )

        print()

    time_elapsed = time.time() - since
    print(
        f'Training complete in '
        f'{time_elapsed // 60:.0f}m '
        f'{time_elapsed % 60:.0f}s'
    )
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

if __name__ == "__main__":
    try:
        dataloaders, class_names, dataset_size = load_data(
            DATA_DIR
        )

        model_ft = initialize_model(
            len(class_names)
        )

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(
            model_ft.parameters(),
            lr=LEARNING_RATE
        )

        model_ft, hist = train_model(
            model_ft,
            dataloaders,
            criterion,
            optimizer_ft,
            num_epochs=NUM_EPOCHS
        )

        torch.save(
            model_ft.state_dict(),
            "fruit_freshness_model.pth"
        )
        print(
            "Model saved as fruit_freshness_model.pth"
        )

        plt.title(
            "Validation Accuracy over Epochs"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(
            range(1, NUM_EPOCHS + 1),
            hist
        )
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        print(
            "Please check if DATA_DIR is correct "
            "and contains subfolders for classes."
        )
