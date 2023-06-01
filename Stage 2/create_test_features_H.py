import torch
from torchvision import datasets

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

import open_clip

from tqdm import tqdm

import numpy as np

import os
from PIL import Image
import pickle

np.random.seed(0)
torch.manual_seed(0)

batch_size = 64

class TestDataset(Dataset):
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

test_dir = 'compressed_dataset/test'
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = 'compressed_dataset/train'

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')

# Turn off gradient calculations for all parameters
for param in model.parameters():
    param.requires_grad = False

model = model.to(device)

print("CLIP loaded.")

train_dataset = datasets.ImageFolder(data_dir, transform=preprocess_train)
class_to_idx = train_dataset.class_to_idx
initial_classes = list(class_to_idx.keys())
classes = ['']*48
for i in range(len(initial_classes)):
    classes[i] = 'a computer generated image of a ' + initial_classes[i]
print("Tokenizer class names :\n", classes)
text = tokenizer(classes).to(device)

test_dataset = TestDataset(test_dir, test_transform=preprocess_val)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

def save_features(file_name, data_loader, image_names=False):
    model.eval()
    print("Computing the features of the images ...")
    loader_encoded = []
    for images, names in tqdm(data_loader):
        images = images.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(text)
            if image_names:
                loader_encoded.append((image_features, text_features, names))
            else:
                loader_encoded.append((image_features, text_features))

    # Save the data to a file
    with open(file_name, 'wb') as f:
        pickle.dump(loader_encoded, f)
    print(f"Image features saved in {file_name}.")

save_features("encoded_test_images_H.pkl", test_loader, image_names=True)