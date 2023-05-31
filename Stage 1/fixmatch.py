import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import os 
from PIL import Image
from torch.optim.lr_scheduler import StepLR

from collections import defaultdict

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dir = 'test'
val_dataset = datasets.ImageFolder(test_dir, transform=transform)
data = DataLoader(val_dataset, batch_size=1, shuffle=True)

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# Data transforms
data_transforms_weak = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transforms_strong = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transforms_strong2 = transforms.Compose([
    transforms.Resize(224),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transforms_strong3 = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transforms_strong4 = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

num_epochs = 100
batch_size = 20 # Must divide the len of the train loader (train_dataset_midjourney contains 460 images and the standard dataset 150)
lr = 0.001
mu = 7 # coefficient of unlabeled batch size

# pseudo label threshold
threshold = 0.95

lambda_u = 1 # coefficient of unlabeled loss
T = 1 # Pseudo-label temperature

data_dir = 'train'
data_dir_midjourney = 'midjourney_train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We need to define a custom Dataset, as the ImageFolder requires folders with the name of the classes
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

# Load the labeled dataset
train_dataset_midjourney = datasets.ImageFolder(data_dir_midjourney, transform=transform)
# train_dataset_empty = datasets.ImageFolder(data_dir, transform=transform)
train_dataset_weak = datasets.ImageFolder(data_dir, transform=data_transforms_weak)
# train_dataset_strong = datasets.ImageFolder(data_dir, transform=data_transforms_strong)
# train_dataset_strong2 = datasets.ImageFolder(data_dir, transform=data_transforms_strong2)
# train_dataset_strong3 = datasets.ImageFolder(data_dir, transform=data_transforms_strong3)
train_dataset_strong4 = datasets.ImageFolder(data_dir, transform=data_transforms_strong4)
train_dataset = [train_dataset_weak, train_dataset_strong4]

# Load the unlabeled dataset
unlabeled_data_dir = 'unlabelled'
unlabeled_dataset = UnlabeledDataset(unlabeled_data_dir, transform_strong=data_transforms_strong, transform_weak=data_transforms_weak)

model = models.resnet50(pretrained=True)
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
num_classes = 10
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)


# Define the optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=lr)


for epoch in range(num_epochs):
    model.train()
    
    train_loader = DataLoader(ConcatDataset(train_dataset + [train_dataset_midjourney]), batch_size=batch_size, shuffle=True, num_workers=4)
    train_iter = iter(train_loader)

    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size*mu, shuffle=True, num_workers=4)
    unlabeled_iter = iter(unlabeled_loader)

    for batch_idx in range(len(train_iter)):
        inputs_x, targets_x = train_iter.next()
        inputs_u_w, inputs_u_s, _ = unlabeled_iter.next()

        inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*mu+1).to(device)
        targets_x = targets_x.to(device)
        logits = model(inputs)
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

        # Backpropagate the gradients
        loss.backward()
        optimizer.step()

        # Print training progress
        print("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Lx: {:.4f}, Lu: {:.4f}".format(
            epoch + 1, num_epochs, batch_idx + 1, len(train_iter), loss.item(), Lx.item(), Lu.item()))
            
    if epoch % 2 == 0:
        score = 0
        i = 0
        class_scores = defaultdict(int)
        class_counts = defaultdict(int)

        model.eval()
        for inputs, label in data:
            i += 1
            with torch.no_grad():
                output = model(inputs.to(device))
            _, predicted = torch.max(output.data, 1)
            class_index = predicted.item()
            class_name = None

            if class_index == 0:
                class_name = 'Conestoga wagon'
            elif class_index == 1:
                class_name = 'bat'
            elif class_index == 2:
                class_name = 'carbine'
            elif class_index == 3:
                class_name = 'cupola'
            elif class_index == 4:
                class_name = 'gosling'
            elif class_index == 5:
                class_name = 'hammer'
            elif class_index == 6:
                class_name = 'peahen'
            elif class_index == 7:
                class_name = 'squash racket'
            elif class_index == 8:
                class_name = 'tragopan'
            elif class_index == 9:
                class_name = 'zinfandel'
            
            if class_index == label[0]:
                score += 1
                class_scores[class_name] += 1
            
            class_counts[class_name] += 1
        print(f"Overall accuracy: {score/i}")

        for class_name in class_counts:
            print(f"Predicted {class_counts[class_name]} {class_name}, {class_scores[class_name]} rights")


torch.save(model.state_dict(), f'./fixmatch.pt')
