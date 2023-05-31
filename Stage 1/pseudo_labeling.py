# Basic pseudo labeling
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset
from torchvision import datasets, models, transforms
from sklearn.model_selection import KFold
import numpy as np
import os 
from PIL import Image

data_dir = 'dataset/train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We need to define a custom Dataset, as the ImageFolder requires folders with the name of the classes
class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, -1


# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
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

num_epochs = 100
k_folds = 5
batch_size = 64
lr = 0.001
subset_size = 10000 # For the unlabelled images
initial_threshold = 0.6
threshold_increment = 0.1
threshold_update_interval = 10
max_threshold = 0.95

# Initialize WandB
wandb.init(project='ResNet_pseudo_labeling', entity='inf473v', config={"learning_rate": lr,
                                                                         "architecture": "ResNet",
                                                                         "epochs": num_epochs,
                                                                         "batch_size": batch_size,
                                                                         "nb_folds": k_folds,
                                                                         "initial_threshold":initial_threshold,
                                                                         "subset_size":subset_size,
                                                                         "unfrozen_layer": "last_one",
                                                                        })

# Set up k-fold cross-validation
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Load the labeled dataset
train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val'])

# Load the unlabeled dataset
unlabeled_data_dir = 'dataset/unlabelled'
unlabeled_dataset = UnlabeledDataset(unlabeled_data_dir, transform=data_transforms['train'])
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_acc_history = []

def get_unlabeled_subset(unlabeled_dataset, subset_size):
    total_size = len(unlabeled_dataset)
    indices = np.random.choice(np.arange(total_size), size=subset_size, replace=False)
    return Subset(unlabeled_dataset, indices)

# Pseudo-labeling function
def generate_pseudo_labels(model, data_loader, device, threshold):
    print("Generating pseudo labels ...")
    model.eval()
    pseudo_labels = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            max_probabilities, max_indices = torch.max(probabilities, dim=1)

            for input, max_probability, max_index in zip(inputs, max_probabilities, max_indices):
                if max_probability.item() > threshold:
                    # Move the tensors back to the CPU before appending to the list
                    input = input.to('cpu')
                    max_index = max_index.to('cpu')
                    pseudo_labels.append((input, torch.tensor([max_index.item()], dtype=torch.long)))

    model.train()
    return pseudo_labels

for fold, (train_indices, val_indices) in enumerate(kfold.split(train_dataset)):
    print(f'Fold {fold + 1}/{k_folds}')

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = models.resnet50(pretrained=True)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    num_classes = 10
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # TODO : Maybe try to unfreeze a few more layers 
    # Unfreeze the last two residual blocks (Layer4)
    # for param in model.layer4.parameters():
    #     param.requires_grad = True

    # Watch the model with WandB
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0

        # Generate pseudo labels for the unlabeled data
        if epoch % 5 == 0:
            threshold = min(initial_threshold + (epoch // threshold_update_interval) * threshold_increment, max_threshold)
            unlabeled_subset = get_unlabeled_subset(unlabeled_dataset, subset_size)
            unlabeled_subset_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=True, num_workers=4)
            pseudo_labels = generate_pseudo_labels(model, unlabeled_subset_loader, device, threshold=threshold)

            if pseudo_labels:
                print(f"Pseudo-labeled samples in epoch {epoch+1}: {len(pseudo_labels)}")

                # Separate the inputs and labels into two tensors
                pseudo_inputs, pseudo_labels = zip(*pseudo_labels)
                pseudo_inputs = torch.stack(pseudo_inputs)
                pseudo_labels = torch.tensor(pseudo_labels)
                pseudo_labeled_dataset = TensorDataset(pseudo_inputs, pseudo_labels)
                def collate_fn(batch):
                    inputs, labels = zip(*batch)
                    inputs = torch.stack(inputs)
                    labels = torch.tensor(labels, dtype=torch.long)
                    return inputs, labels

                train_loader = DataLoader(
                    ConcatDataset([train_dataset, pseudo_labeled_dataset]),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    collate_fn=collate_fn,
                )

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Log training loss and accuracy to WandB
        wandb.log({'train_loss': epoch_loss, 'train_accuracy': epoch_acc})

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        for val_inputs, val_labels in val_loader:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                val_outputs = model(val_inputs)
                _, val_preds = torch.max(val_outputs, 1)
                val_loss = criterion(val_outputs, val_labels)

            val_running_loss += val_loss.item() * val_inputs.size(0)
            val_running_corrects += torch.sum(val_preds == val_labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.float() / len(val_loader.dataset)

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        # Log validation loss and accuracy to WandB
        wandb.log({'val_loss': val_epoch_loss, 'val_accuracy': val_epoch_acc})

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc

    print(f'Best val Acc: {best_val_acc:.4f}')
    val_acc_history.append(best_val_acc.item())

    torch.save(model.state_dict(), './pseudo_labeling.pt')

print(f'Mean validation accuracy across {k_folds}-fold cross-validation (optimistic): {np.mean(val_acc_history):.4f}')

# Close the WandB run
wandb.finish()