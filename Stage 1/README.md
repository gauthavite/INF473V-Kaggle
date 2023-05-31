# Stage 2 Competition
In this folder, we use standard methods for semi supervised learning, with standard models built for ImageNet to tacke our problem.

## Getting started 
The dataset should be contained in a folder `dataset`, with three subfolders `train`, `unlabelled`, and `test`.

There isn't any specific import required.

## How to use
Running 
```bash 
python fixmatch.py
```
will create a pytorch file `fixmatch.pt` of the fine-tuned model.

You can then run 
```bash
python create_submission.py
```
to create the submission.