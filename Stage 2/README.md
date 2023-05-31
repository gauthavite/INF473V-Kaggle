# Stage 2 Competition
In this folder, we leverage the zero-shot capabilities of the CLIP Model for Weakly Supervised classification on a synthetic dataset.

## Getting started
The dataset should be contained in a folder named `compressed_dataset`.
You also need to install OpenCLIP with : 
```bash
pip install open_clip_torch
```

## How to use
Due to the size of the OpenCLIP model we are using, the process is a bit meticulous. 

We create two different models, and then we apply an ensemble method by avering the probabilities of the two models. 

### First Model
You can create the first model by typing the following command, which will finetune a linear layer at the end of the OpenCLIP model and save it in the file `open_clip_fixmatch.pt`
```bash
python open_clip_fixmatch.py
```

### Second Model
Maxence ?

### Creating the submission 
First, you need to compute the features on the test dataset by typing
```bash 
python create_test_features.py
```
This step should take about 20 minutes.

Then, you can run 
```bash
python create_open_clip_ensemblist_submission.py
```
which will create the Kaggle submission by avering the probabilities of the two models.