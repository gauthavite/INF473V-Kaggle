import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

from sklearn.model_selection import StratifiedShuffleSplit

from torch.optim import Adam, SGD, lr_scheduler

import clip

import numpy as np

num_epochs = 20
batch_size = 64

data_dir = 'compressed_dataset/train'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device,jit=False)
model.float()

print("CLIP loaded")

# Set up the transforms and dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms["train"])
val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms["val"])

class_to_idx = train_dataset.class_to_idx
classes = list(class_to_idx.keys())
print(classes)
text = clip.tokenize(classes).to(device)

# get the labels of the images
labels = [label for _, label in train_dataset]
# create a StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# get the indices for the training set and the test set
train_indices, test_indices = next(sss.split(labels, labels))
# create SubsetRandomSampler objects using the indices
train_dataset = Subset(train_dataset, train_indices)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
val_dataset = Subset(val_dataset, test_indices)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

print("Data loaders created")

# Set up the optimizer
optimizer = Adam(model.parameters(), lr=5e-7)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)

# Set up the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# List to store the validation accuracies
val_accuracies = []

# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(images, text)

        # Compute loss
        loss = loss_fn(logits_per_image, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_predicted = []
        all_labels = []
        class_correct = list(0. for _ in range(len(classes)))
        class_total = list(0. for _ in range(len(classes)))

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, text)

            # Compute probabilities
            probs = logits_per_image.softmax(dim=-1)

            # Get predictions
            _, predicted = torch.max(probs, 1)

            # Update total and correct count
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predicted labels and true labels
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate correct predictions per class
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):  # For each image in batch
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        accuracy = correct / total

        val_accuracy = []
        for i in range(len(classes)):
            val_accuracy.append(class_correct[i] / class_total[i])
        val_accuracies.append(val_accuracy)

        # Calculate average accuracy for each class
        if epoch % 5 == 0:
            avg_val_accuracy = np.mean(val_accuracies, axis=0)
            for i in range(len(classes)):
                print('Average accuracy of %5s over all epochs: %2d %%' % (classes[i], 100 * avg_val_accuracy[i]))

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}')