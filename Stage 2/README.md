# Stage 2 Competition
In this folder, we leverage the zero-shot capabilities of the CLIP Model for Weakly Supervised classification on a synthetic dataset.


## Getting started
The dataset should be contained in a folder named `compressed_dataset`.
You also need to install OpenCLIP with : 
```bash
pip install open_clip_torch
```


## How to use
We explain how to obtain the csv submission file for Kaggle, with two different models.


## First Model : CLIP fine-tuned with FixMatch 
You can create the first model by typing the following command, which will fine-tune a linear layer at the end of the OpenCLIP model and save it in the file `open_clip_fixmatch.pt`
```bash
python open_clip_fixmatch.py
```

First, you need to compute the features on the test dataset by typing
```bash 
python create_test_features.py
```
This step should take about 20 minutes.

Then, you can run 
```bash
python create_open_clip_submission.py
```
which will create the Kaggle submission for the FixMatch model.


## Second Model : CLIP fine-tuned with pseudo labeling and weighted function
You can directly create the submission of this model by typing 
```bash
python open_clip_pseudo_weights.py
```
It will also save the linear layer in the file `fc_linear_test.pt`.