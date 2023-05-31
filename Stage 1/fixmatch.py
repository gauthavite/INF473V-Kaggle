import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import numpy as np
import os 
from PIL import Image
import PIL
import random
from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import Sampler

print('import ok')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####HYPERPARAMETERS######
num_epochs = 20
batch_size = 60
lr = 0.001
mu = 7 # coefficient of unlabeled batch size
threshold = 0.95 # pseudo label threshold
lambda_u = 1 # coefficient of unlabeled loss
T = 1 # Pseudo-label temperature

PARAMETER_MAX = 10

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

size=224

num_classes = 10

data_dir = 'dataset/train'
unlabeled_data_dir = 'dataset/unlabelled'


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class SameSeedSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = list(range(len(data_source)))
        np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            # (Color, 1.8, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs

class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img


class TransformFixMatchStrong(object):
    def __init__(self, mean, std,n,m):
        self.strong = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=n, m=m)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    def __call__(self, x):
        strong = self.strong(x)
        return self.normalize(strong)

class TransformFixMatchWeak(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomAffine(degrees=0, translate=(0.125,0.125)),
            transforms.RandomHorizontalFlip(),
            ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    def __call__(self, x):
        weak = self.weak(x)
        return self.normalize(weak)

val_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    
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


#Create DATASETS
# Create the labeled dataset
train_dataset = datasets.ImageFolder(data_dir, transform=TransformFixMatchWeak(mean=mean,std=std))
val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

# get the labels of the images
labels = [label for _, label in train_dataset]
# create a StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# get the indices for the training set and the test set
train_indices, test_indices = next(sss.split(labels, labels))
# create SubsetRandomSampler objects using the indices
train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, test_indices)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Create the unlabeled dataset
unlabeled_dataset = UnlabeledDataset(unlabeled_data_dir, transform_strong=TransformFixMatchStrong(mean=mean,std=std,n=5,m=10), transform_weak=TransformFixMatchWeak(mean=mean,std=std))
print('datasets created')

def get_unlabeled_subset(unlabeled_dataset, subset_size):
    total_size = len(unlabeled_dataset)
    indices = np.random.choice(np.arange(total_size), size=subset_size, replace=False)
    return Subset(unlabeled_dataset, indices)

#Create MODEL
#Try to change to a ViT
model = models.vit_b_16(pretrained=True)
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.heads.head.in_features
model.heads.head = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

print('ok model')

# Define the optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_iter = iter(train_loader)

    unlabeled_loader = DataLoader(get_unlabeled_subset(unlabeled_dataset, len(train_iter)*mu*batch_size), batch_size=batch_size*mu, shuffle=True, num_workers=4)
    unlabeled_iter = iter(unlabeled_loader)

    for batch_idx in range(len(train_iter)):
        inputs_x, targets_x = next(train_iter)
        inputs_u_w, inputs_u_s, _ = next(unlabeled_iter)

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

        Lu = (F.cross_entropy(logits_u_s, targets_u,
                                reduction='none') * mask).mean()

        loss = Lx + lambda_u * Lu


        optimizer.zero_grad()

        # Backpropagate the gradients
        loss.backward()
        optimizer.step()
        
        # Print training progress
        print("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Lx: {:.4f}, Lu: {:.4f}".format(
            epoch + 1, num_epochs, batch_idx + 1, len(train_iter), loss.item(), Lx.item(), Lu.item()))
        
    # Validation step
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0

    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        val_running_loss += loss.item() * inputs.size(0)
        val_running_corrects += torch.sum(preds == labels.data)

    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_epoch_acc = val_running_corrects.float() / len(val_loader.dataset)

    print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

torch.save(model.state_dict(), './fixmatch.pt')