import glob
import os

from matplotlib import pyplot as plt
import numpy as np
import open_clip
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm

import wandb

N_SPLITS = 5
BATCH_SIZE = 32
LR = 1e-03
DEVICE = "cuda"
MODEL = "ViT-H-14"
NUM_CLASSES = 48
NUM_EPOCHS = 100
ROOT_DIR = "compressed_dataset"
PRETRAINED = "laion2b_s32b_b79k"
TEXT_TEMPLATE = lambda c: f"a computer generated image of a {c}"
THRESHOLD = 0.99

# Initialize WandB
wandb.init(project='OpenClip Maxence', entity='inf473v', config={"architecture": "ViT-H-14",
                                                                "epochs": NUM_EPOCHS,
                                                                "batch_size": BATCH_SIZE,
                                                                "lr": LR,
                                                                "nb_folds": N_SPLITS,
                                                                "threshold":THRESHOLD,
                                                                })

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL)

model = model.to(DEVICE)


labeled_dataset = torchvision.datasets.ImageFolder(os.path.join(ROOT_DIR, "train"), transform=preprocess)
labeled_dataloader = torch.utils.data.DataLoader(labeled_dataset, batch_size=BATCH_SIZE, num_workers=4)

classes = labeled_dataset.classes
text_inputs = tokenizer(map(TEXT_TEMPLATE, classes)).to(DEVICE)

def get_features(dataloader):
    all_features = []
    all_labels = []
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, labels in tqdm(dataloader):
            features = model.encode_image(images.to(DEVICE))
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def step_epoch_train(pl_dataloader, loss_fn):
    running_loss = 0.0
    pbar = tqdm(pl_dataloader, leave=False)
    n = len(pl_dataloader)
    step = 0
    for inputs, labels in pbar:
        step += 1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = linear_module(inputs.to(torch.float32))
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            pbar.set_description(f"train batchCE: {loss.item()}", refresh=True)
        running_loss += loss.item()
    running_loss /= n
    return running_loss


def step_epoch_eval(dataloader, loss_fn, val_loss_fn):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad(), torch.cuda.amp.autocast():
        val_loss = 0
        val_loss_nw = 0
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # calculate outputs by running images through the network
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = linear_module(images)
                val_loss += loss_fn(outputs, labels).item()
                val_loss_nw += val_loss_fn(outputs, labels).item()
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_loss /= len(dataloader)
        val_loss_nw /= len(dataloader)
    return val_loss, val_loss_nw, correct / total


class UnlabeledDataset:
    def __init__(self, root_dir, preprocess):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.filepaths = glob.glob(f"{self.root_dir}/*.jpg")
    
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        return filepath, self.preprocess(Image.open(filepath))
    
    def __len__(self):
        return len(self.filepaths)
    
    
class PseudoLabeledDataset:
    def __init__(self, unlabeled_dataloader, preprocess):
        self.unlabeled_dataloader = unlabeled_dataloader
        self.files = []
        self.targets = []
        self.preprocess = preprocess
        self.init()
        
    def init(self):
        print("Initialisation of pseudo-labeled dataset.")
        for batch in tqdm(unlabeled_dataloader):
            filepaths, images = batch
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(images.to(DEVICE))
                text_features = model.encode_text(text_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            mask = torch.squeeze((torch.max(text_probs, dim=1)[0] > THRESHOLD)).cpu().numpy()
            self.files.extend(np.array(filepaths)[mask])
            self.targets.extend(map(lambda x: x.item(), torch.argmax(text_probs, dim=1)[mask]))
    
    def __getitem__(self, idx):
        filepath = self.files[idx]
        return self.preprocess(Image.open(filepath)), self.targets[idx]
    
    def __len__(self):
        return len(self.files)
    

unlabeled_dataset = UnlabeledDataset(f"{ROOT_DIR}/unlabelled/", preprocess)   
unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=256, num_workers=4)
pseudo_labeled_dataset = PseudoLabeledDataset(unlabeled_dataloader, preprocess)

# get the labels of the images
labels = [label for _, label in labeled_dataset]
# create a StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=42)
accuracies = []

for i, (train_indices, test_indices) in enumerate(sss.split(labels, labels)):
    print(f"Fold {i+1}")
    # create SubsetRandomSampler objects using the indices
    train_dataset = torch.utils.data.Subset(labeled_dataset, train_indices)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_dataset = torch.utils.data.Subset(labeled_dataset, test_indices)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)


    train_features, train_labels = get_features(train_dataloader)
    test_features, test_labels = get_features(test_dataloader)

    featured_train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    featured_train_dataloader = torch.utils.data.DataLoader(featured_train_dataset, batch_size=BATCH_SIZE)
        
    featured_test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    featured_test_dataloader = torch.utils.data.DataLoader(featured_test_dataset, batch_size=BATCH_SIZE)


    pl_dataset = torch.utils.data.ConcatDataset([train_dataset, pseudo_labeled_dataset])
    pl_dataloader = torch.utils.data.DataLoader(pl_dataset, batch_size=BATCH_SIZE)


    y = []
    for _, x in tqdm(pl_dataloader):
        y.extend(x)

    hist = np.histogram(y, bins=np.arange(0, 49, 1), density=True)
    classes = np.array(classes)
    sorted_indices = np.argsort(-hist[0])
    weights = 1 / hist[0]


    pl_features, pl_labels = get_features(pl_dataloader)
    pl_dataset = torch.utils.data.TensorDataset(pl_features, pl_labels)
    pl_dataloader = torch.utils.data.DataLoader(pl_dataset, batch_size=BATCH_SIZE)


    in_features = 1024
    out_features = NUM_CLASSES

    linear_module = nn.Sequential(nn.Linear(in_features, out_features), nn.Softmax(dim=1)).to(DEVICE)

    # Watch the model with WandB
    wandb.watch(linear_module)

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(DEVICE))
    val_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_module.parameters(), lr=LR)


    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        linear_module.train()
        pl_dataloader = torch.utils.data.DataLoader(pl_dataset, batch_size=BATCH_SIZE, shuffle=True)
        running_loss = step_epoch_train(pl_dataloader, loss_fn)

        linear_module.eval()
        val_loss, val_loss_nw, acc = step_epoch_eval(featured_test_dataloader, loss_fn, val_loss_fn)
        print(f"[{epoch + 1}] tr_loss: {running_loss:.4f} val_loss: {val_loss:.4f} "
            f"val_loss_nw: {val_loss_nw:.4f} acc: {100 * acc:.2f}")

        # Log training loss and accuracy to WandB
        wandb.log({'training loss': running_loss, 'val_loss': val_loss, 'val_accuracy': acc})
    accuracies.append(acc)

print("Average accuracy over", N_SPLITS, "folds :", np.array(accuracies).mean())
print("Standard deviation of accuracy over", N_SPLITS, "folds :", np.array(accuracies).std())

torch.save(linear_module, "fc_linear_test.pt")

### Creating the submission ###

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)
    
exam_dataset = TestDataset(f"{ROOT_DIR}/test/", preprocess)  
exam_dataloader = torch.utils.data.DataLoader(exam_dataset, batch_size=BATCH_SIZE)

with open("submission_pseudo_weighted.csv", "w") as f:
    f.write("id,label\n")
    for images, names in tqdm(exam_dataloader):
        images = images.to(DEVICE)        
        linear_module.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            images = model.encode_image(images)
            images /= images.norm(dim=-1, keepdim=True)
            outputs = linear_module(images)
            _, predicted = torch.max(outputs, 1)
        for i in range(len(names)):
            f.write(f"{names[i]},{classes[predicted[i]]}\n")