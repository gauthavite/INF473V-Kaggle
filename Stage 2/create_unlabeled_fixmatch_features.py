import torch
from torchvision import datasets

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

import open_clip

from tqdm import tqdm

import numpy as np

import os
from PIL import Image
import pickle

np.random.seed(0)
torch.manual_seed(0)

batch_size = 64*5
unlabeled_subset_size = 4032

device = "cuda" if torch.cuda.is_available() else "cpu"
unlabeled_dir = 'compressed_dataset/unlabelled'
data_dir = 'compressed_dataset/train'

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

# Turn off gradient calculations for all parameters
for param in model.parameters():
    param.requires_grad = False

model = model.to(device)

print("CLIP loaded.")

train_dataset = datasets.ImageFolder(data_dir, transform=preprocess_train)
class_to_idx = train_dataset.class_to_idx
classes = list(class_to_idx.keys())
# for i in range(len(classes)):
#     classes[i] = 'a stable diffusion generated image of a ' + classes[i]
print("Tokenizer class names :\n", classes)
text = tokenizer(classes).to(device)

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
unlabeled_dataset = UnlabeledDataset(unlabeled_dir, transform_strong=preprocess_train, transform_weak=preprocess_val)
indices = np.random.choice(np.arange(len(unlabeled_dataset)), size=unlabeled_subset_size, replace=False)
unlabeled_subset = Subset(unlabeled_dataset, indices)
unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, num_workers=4)


def save_features(file_name, data_loader):
    model.eval()
    print("Computing the features of the images ...")
    loader_encoded = []
    for image_weak, image_strong, _ in tqdm(data_loader):
        image_weak, image_strong = image_weak.to(device), image_strong.to(device)
        with torch.no_grad():
            image_weak_features = model.encode_image(image_weak)
            image_strong_features = model.encode_image(image_strong)
            text_features = model.encode_text(text)
            loader_encoded.append((image_weak_features, image_strong_features, text_features))

    # Save the data to a file
    with open(file_name, 'wb') as f:
        pickle.dump(loader_encoded, f)
    print(f"Image features saved in {file_name}.")


save_features("encoded_unlabeled_fixmatch.pkl", unlabeled_loader)
