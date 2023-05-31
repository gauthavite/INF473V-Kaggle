from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
from torchvision import transforms, models
import torch
import torch.nn as nn

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #Metal framework for Apple Silicon support

data_transforms_eval = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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


test_path = "test"
train_path = "train"
test_loader = DataLoader(TestDataset(test_path, data_transforms_eval), batch_size=32, shuffle=False)

# Load model and checkpoint
model_path = 'fixmatch.pt'
model = models.resnet50(pretrained=True)
num_classes = 10
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.to(device)

class_names = sorted(os.listdir(train_path))

# Create submission.csv
submission = pd.DataFrame(columns=["id", "label"])

for i, batch in enumerate(test_loader):
    images, image_names = batch
    images = images.to(device)
    preds = model(images)
    preds = preds.argmax(1)
    preds = [class_names[pred] for pred in preds.cpu().numpy()]
    submission = pd.concat(
        [
            submission,
            pd.DataFrame({"id": image_names, "label": preds}),
        ]
    )
    print(i, "/", len(test_loader))
submission.to_csv(f"submission.csv", index=False)
