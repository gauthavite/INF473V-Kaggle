from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torch


from tqdm import tqdm

import pickle

batch_size = 64
test_path = "compressed_dataset/test"


device = "cuda" if torch.cuda.is_available() else "cpu"
fc = torch.nn.Linear(1280,1280).to(device)
fc.load_state_dict(torch.load("./open_clip_fixmatch.pt", map_location=torch.device('cpu')))
fc = fc.to(device)

initial_classes = ['Conestoga wagon', 'Entoloma lividum', 'Habenaria_bifolia', 'Rhododendron viscosum', 'Rhone wine', 'Salvelinus fontinalis', 'bat', 'bearberry', 'black-tailed deer ', 'brick red', 'carbine', 'ceriman', 'control room', 'cotton candy', 'couscous', 'cupola', 'damask violet', 'digital subscriber line', 'drawing room', 'duckling', 'ethyl alcohol', 'flash', 'florist', 'floss', 'gift shop', 'gosling', 'grenadine', 'guava', 'hammer', 'kingfish', 'organ loft', 'peahen', 'pinwheel', 'platter', 'plunge', 'shovel', 'silkworm', 'snowboard', 'spiderwort', 'squash racket', 'steering wheel', 'swamp chestnut oak', 'toadstool', 'tragopan', 'veloute', 'vintage', 'waldmeister', 'zinfandel']

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


test_file_name = "encoded_test_images_with_prompts.pkl"
if os.path.isfile(test_file_name):
    # Load the data from the file
    with open(test_file_name, 'rb') as f:
        test_loader = pickle.load(f)
else:
    raise FileNotFoundError

# Create submission.csv
submission = pd.DataFrame(columns=["id", "label"])

# model.eval()
fc.eval()
for image_features, text_features, image_names in tqdm(test_loader):

    with torch.no_grad():
        image_features = fc(image_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        _, predicted = torch.max(text_probs, 1)

        preds = [initial_classes[pred] for pred in predicted.cpu().numpy()]
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "label": preds}),
            ]
        )
submission.to_csv(f"submission_open_clip_fixmatch.csv", index=False)