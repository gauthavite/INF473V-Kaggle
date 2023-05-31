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

fc_without_prompts = torch.nn.Linear(1280,1280).to(device)
fc_without_prompts.load_state_dict(torch.load("./open_clip_finetuning.pt", map_location=torch.device('cpu')))
fc_without_prompts = fc_without_prompts.to(device)

fc_with_prompts = torch.nn.Linear(1280,1280).to(device)
fc_with_prompts.load_state_dict(torch.load("./open_clip_finetuning_prompts.pt", map_location=torch.device('cpu')))
fc_with_prompts = fc_with_prompts.to(device)

classes = ['Conestoga wagon', 'Entoloma lividum', 'Habenaria_bifolia', 'Rhododendron viscosum', 'Rhone wine', 'Salvelinus fontinalis', 'bat', 'bearberry', 'black-tailed deer ', 'brick red', 'carbine', 'ceriman', 'control room', 'cotton candy', 'couscous', 'cupola', 'damask violet', 'digital subscriber line', 'drawing room', 'duckling', 'ethyl alcohol', 'flash', 'florist', 'floss', 'gift shop', 'gosling', 'grenadine', 'guava', 'hammer', 'kingfish', 'organ loft', 'peahen', 'pinwheel', 'platter', 'plunge', 'shovel', 'silkworm', 'snowboard', 'spiderwort', 'squash racket', 'steering wheel', 'swamp chestnut oak', 'toadstool', 'tragopan', 'veloute', 'vintage', 'waldmeister', 'zinfandel']

test_file_name_without_prompts = "encoded_test_images.pkl"
if os.path.isfile(test_file_name_without_prompts):
    # Load the data from the file
    with open(test_file_name_without_prompts, 'rb') as f:
        test_loader_without_prompts = pickle.load(f)
else:
    raise FileNotFoundError

test_file_name_with_prompts = "encoded_test_images_with_prompts.pkl"
if os.path.isfile(test_file_name_with_prompts):
    # Load the data from the file
    with open(test_file_name_with_prompts, 'rb') as f:
        test_loader_with_prompts = pickle.load(f)
else:
    raise FileNotFoundError

# Create submission.csv
submission = pd.DataFrame(columns=["id", "label"])

fc_with_prompts.eval()
fc_without_prompts.eval()
for (image_features, text_features, image_names), (image_features_with, text_features_with, image_names_with) in tqdm(zip(test_loader_without_prompts, test_loader_with_prompts)):
    for i in range(len(image_names)):
        assert(image_names[i] == image_names_with[i])
        
    with torch.no_grad():
        image_features = fc_without_prompts(image_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        image_features_with = fc_with_prompts(image_features_with)
        image_features_with /= image_features_with.norm(dim=-1, keepdim=True)
        text_features_with /= text_features_with.norm(dim=-1, keepdim=True)
        text_probs_with = (100.0 * image_features_with @ text_features_with.T).softmax(dim=-1)

        _, predicted = torch.max(text_probs + text_probs_with, 1)

        preds = [classes[pred] for pred in predicted.cpu().numpy()]
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "label": preds}),
            ]
        )
submission.to_csv(f"submission_open_clip_ensembliste.csv", index=False)