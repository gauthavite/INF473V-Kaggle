import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import random
from PIL import Image



class MixupTransform:
    def __init__(self, dataset, alpha):
        self.dataset = dataset
        self.alpha = alpha

    def __call__(self, x):
        lam = torch.tensor(random.betavariate(self.alpha, self.alpha)).float()
        index = torch.randperm(len(self.dataset))
        x2 = self.dataset[index[0]][0]
        mixed_x = lam * x + (1 - lam) * x2
        return mixed_x

def mixup_data(dataset, alpha, num_mixes):
    mixup_transform = MixupTransform(dataset, alpha)
    mixup_dataset = []
    for i in range(num_mixes):
        index = random.randint(0, len(dataset) - 1)
        x = dataset[index][0]
        mixed_x = mixup_transform(x)
        mixup_dataset.append((mixed_x, dataset[index][1]))
    return mixup_dataset


# Load original dataset
original_dataset = datasets.ImageFolder('./train/bat', transform=transforms.ToTensor())

# Create mixup dataset
alpha = 1.0 # mixup hyperparameter
num_mixes = 1000 # number of mixup images to generate per original image
mixup_dataset = mixup_data(original_dataset, alpha, num_mixes)





# define the transform to convert tensor images to PIL images
to_pil = transforms.ToPILImage()

# loop over the mixup dataset and save each tensor image as a PNG file
for i, (image, label) in enumerate(mixup_dataset):
    # convert tensor image to PIL image
    pil_image = to_pil(image)
    # save PIL image as a PNG file
    pil_image.save(f"./train/bat_mixup/mixup_{i}.png")



