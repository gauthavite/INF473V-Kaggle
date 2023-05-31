import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"


batch_size = 64

train_file_name_with_prompts = "encoded_train_images_ju_fold_5.pkl"
if os.path.isfile(train_file_name_with_prompts):
    with open(train_file_name_with_prompts, 'rb') as f:
        train_loader_with_prompts, val_loader_with_prompts = pickle.load(f)
else:
    raise FileNotFoundError

fc_ju = torch.nn.Linear(1280,1280).to(device)
fc_ju.load_state_dict(torch.load("./open_clip_finetuning_prompts_ju.pt", map_location=torch.device('cpu')))
fc_ju = fc_ju.to(device)

classes = ['Conestoga wagon', 'Entoloma lividum', 'Habenaria_bifolia', 'Rhododendron viscosum', 'Rhone wine', 'Salvelinus fontinalis', 'bat', 'bearberry', 'black-tailed deer ', 'brick red', 'carbine', 'ceriman', 'control room', 'cotton candy', 'couscous', 'cupola', 'damask violet', 'digital subscriber line', 'drawing room', 'duckling', 'ethyl alcohol', 'flash', 'florist', 'floss', 'gift shop', 'gosling', 'grenadine', 'guava', 'hammer', 'kingfish', 'organ loft', 'peahen', 'pinwheel', 'platter', 'plunge', 'shovel', 'silkworm', 'snowboard', 'spiderwort', 'squash racket', 'steering wheel', 'swamp chestnut oak', 'toadstool', 'tragopan', 'veloute', 'vintage', 'waldmeister', 'zinfandel']
for i in range(len(classes)):
    classes[i] = 'a stable diffusion generated image of a ' + classes[i]

# Evaluate on validation set
fc_ju.eval()
with torch.no_grad():
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    for image_features, text_features, labels in val_loader_with_prompts:
       
        image_features = fc_ju(image_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get predictions
        _, predicted = torch.max(text_probs, 1)

        # Update accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Save all the predicted labels and true labels for each batch
        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    
    # Print confusion matrix and classification report
    print('Confusion Matrix:')
    cm = confusion_matrix(all_labels, all_predicted)
    print(cm)
    
    # Customize the color thresholds and corresponding colors as per your requirement
    cmap = plt.cm.viridis

    # Set the maximum and minimum values in the matrix
    vmin = np.min(cm)
    vmax = np.max(cm)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))


    # Plot the matrix values as an image
    im = ax.imshow(cm, cmap=cmap, vmin=vmin, vmax=vmax)

    # Set the ticks and labels for the x-axis and y-axis
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

    # Set the title and axis labels
    ax.set_title("Confusion Matrix of clip", fontsize=12)
    ax.set_xlabel("Predicted Labels", fontsize=10)
    ax.set_ylabel("True Labels", fontsize=10)

    # Adjust the layout to show the full image
    plt.tight_layout()

    # Save the figure with higher resolution
    plt.savefig('confusion_matrix.png', dpi=300)


    print('Classification Report:')
    print(classification_report(all_labels, all_predicted, target_names=classes))

