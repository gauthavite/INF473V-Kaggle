import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets

from sklearn.model_selection import StratifiedShuffleSplit

from torch.optim import Adam, lr_scheduler

import torch.nn.functional as F

import clip

import os
from PIL import Image

lr = 5e-7
num_epochs = 100
batch_size = 16
mu = 5 # coefficient of unlabeled batch size
threshold = 0.95 # pseudo label threshold
lambda_u = 1 # coefficient of unlabeled loss
T = 1 # Pseudo-label temperature

data_dir = 'compressed_dataset/train'
unlabeled_data_dir = 'compressed_dataset/unlabelled'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device,jit=False)
model.float()

print("CLIP loaded")

# Set up the transforms and dataset
data_transforms = {
    'train_weak': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'train_strong': transforms.Compose([
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

train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms["train_weak"])
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

class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform_weak=None, transform_strong=None):
        self.root_dir = root_dir
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        image_weak = self.transform_weak(image)
        image_strong = self.transform_strong(image)
        return image_weak, image_strong, -1

# Load the unlabeled dataset
unlabeled_dataset = UnlabeledDataset(unlabeled_data_dir, transform_strong=data_transforms["train_strong"], transform_weak=data_transforms["train_weak"])

print("Data loaders created")

optimizer = Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    train_iter = iter(train_loader)

    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size*mu, shuffle=True, num_workers=4)
    unlabeled_iter = iter(unlabeled_loader)

    for batch_idx in range(len(train_iter)):
        inputs_x, targets_x = train_iter.next()
        inputs_u_w, inputs_u_s, _ = unlabeled_iter.next()

        inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*mu+1).to(device)
        targets_x = targets_x.to(device)
        logits, _ = model(inputs, text)
        logits = de_interleave(logits, 2*mu+1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits

        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach()/T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)

        mask = max_probs.ge(threshold).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

        loss = Lx + lambda_u * Lu

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        print("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Lx: {:.4f}, Lu: {:.4f}".format(
            epoch + 1, num_epochs, batch_idx + 1, len(train_iter), loss.item(), Lx.item(), Lu.item()))

    scheduler.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, text)

            # Compute probabilities
            probs = logits_per_image.softmax(dim=-1)

            # Get predictions
            _, predicted = torch.max(probs, 1)

            # Update accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}')

torch.save(model.state_dict(), './clip_fixmatch.pt')
