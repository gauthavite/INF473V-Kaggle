import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from torch.optim import Adam, lr_scheduler

import torch.nn.functional as F

import os
import pickle

import numpy as np

num_epochs = 100
batch_size = 64
lr = 1e-5
weight_decay = 0
n_splits = 5 # number of "folds"
mu = 7 # coefficient of unlabeled batch size
threshold = 0.95 # pseudo label threshold
lambda_u = 1 # coefficient of unlabeled loss
T = 1 # Pseudo-label temperature

data_dir = 'compressed_dataset/train'

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = datasets.ImageFolder(data_dir)
class_to_idx = train_dataset.class_to_idx
classes = list(class_to_idx.keys())
for i in range(len(classes)):
    classes[i] = 'a stable diffusion generated image of a ' + classes[i]

print(classes)

unlabeled_file_name = "encoded_unlabeled_fixmatch.pkl"
if os.path.isfile(unlabeled_file_name):
    # Load the data from the file
    with open(unlabeled_file_name, 'rb') as f:
        unlabeled_loader = pickle.load(f)
else:
    raise FileNotFoundError

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

accuracies = []
for i in range(n_splits):
    print(f"Fold {i+1}")

    fc = nn.Linear(1280,1280).to(device)
    nn.init.xavier_uniform_(fc.weight)
    fc.weight.data *= 0.5  # Scales the weights by 0.5
    fc.weight.data.copy_(torch.eye(1280))

    optimizer = Adam(fc.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 720*0.8*num_epochs)

    loss_fn = torch.nn.CrossEntropyLoss()

    file_name = f'encoded_train_images_fold_{i+1}.pkl'
    if os.path.isfile(file_name):
        # Load the data from the file
        with open(file_name, 'rb') as f:
            train_loader_encoded, val_loader_encoded = pickle.load(f)
            print(f"Image features retrieved.")
    else:
        raise FileNotFoundError

    # Fine-tuning loop
    for epoch in range(num_epochs):
        fc.train()

        train_iter = iter(train_loader_encoded)
        unlabeled_iter = iter(unlabeled_loader)

        for batch_idx in range(len(train_loader_encoded)):
            image_features, text_features, labels = next(train_iter)

            image_u_w_features, image_u_s_features, text_features_u = [], [], []
            for i in range(mu):
                img_u_w, img_u_s, text_u = next(unlabeled_iter)
                image_u_w_features.append(img_u_w)
                image_u_s_features.append(img_u_s)
                text_features_u.append(text_u)
            image_u_w_features = torch.cat(image_u_w_features, dim=0)
            image_u_s_features = torch.cat(image_u_s_features, dim=0)
            text_features_u = torch.cat(text_features_u, dim=0)

            image_features = fc(image_features)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T)
            Lx = F.cross_entropy(text_probs, labels, reduction='mean')

            image_u_w_features = fc(image_u_w_features)
            image_u_w_features = image_u_w_features/image_u_w_features.norm(dim=-1, keepdim=True)
            image_u_s_features = fc(image_u_s_features)
            image_u_s_features = image_u_s_features/image_u_s_features.norm(dim=-1, keepdim=True)
            text_features_u = text_features_u/text_features_u.norm(dim=-1, keepdim=True)

            text_probs_u_w = (100.0 * image_u_w_features @ text_features_u.T)
            text_probs_u_s = (100.0 * image_u_s_features @ text_features_u.T)

            pseudo_label = torch.softmax(text_probs_u_w.detach()/T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)

            mask = max_probs.ge(threshold).float()

            Lu = (F.cross_entropy(text_probs_u_s, targets_u, reduction='none') * mask).mean()

            loss = Lx + lambda_u * Lu

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate on validation set
        fc.eval()
        correct = 0
        total = 0
        for image_features, text_features, labels in val_loader_encoded:
            with torch.no_grad():
                image_features = fc(image_features)
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)
                text_features = text_features/text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                _, pred = torch.max(text_probs, 1)

                total += labels.size(0)
                correct += (pred == labels).sum().item()

        accuracy = correct / total
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}')

    accuracies.append(accuracy)

print("Average accuracy over", n_splits, "folds :", np.array(accuracies).mean())
print("Standard deviation of accuracy over", n_splits, "folds :", np.array(accuracies).std())

model_save_path = "./open_clip_fixmatch.pt"
print("Saving the last fold model in the file", model_save_path)
torch.save(fc.state_dict(), model_save_path)
