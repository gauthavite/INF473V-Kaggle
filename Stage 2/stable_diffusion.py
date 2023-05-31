import torch
from io import BytesIO
import os
from diffusers import StableDiffusionImg2ImgPipeline
import matplotlib.pyplot as plt
import PIL
from PIL import Image


###ENLEVER LES IMAGES NOIRES QUI PEUVENT ETRE GENEREES LORSQUE L'IMAGE EST CENSUREE

def is_image_all_black(img):
    width, height = img.size
    # Check each pixel to see if it is black
    for y in range(height):
        for x in range(width):
            pixel = img.getpixel((x, y))
            if pixel != (0, 0, 0):  # If the pixel is not black
                return False
    return True

def transform(img,i,prompt):
    im_path = os.path.join(prompt_folder, img)
    print(im_path)
    image = Image.open(im_path)
    images = pipe(prompt=prompt, image=image, strength=0.75, guidance_scale=2).images
    file_name = f"{i}.jpg"
    output_path = os.path.join(prompt_folder, file_name)
    return output_path,images

model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')) #Metal framework for Apple Silicon support
pipe = pipe.to(device)
pipe.enable_attention_slicing()

prompts = ['platter', 'grenadine', 'toadstool', 'gift shop', 'digital subscriber line', 'Conestoga wagon', 'Rhone wine', 'squash racket', 'bearberry', 'couscous', 'peahen', 'Habenaria_bifolia', 'florist', 'tragopan', 'flash', 'shovel', 'guava', 'waldmeister', 'drawing room', 'carbine', 'veloute', 'Entoloma lividum', 'bat', 'damask violet', 'ceriman', 'steering wheel', 'cupola', 'Salvelinus fontinalis', 'spiderwort', 'cotton candy', 'snowboard', 'Rhododendron viscosum', 'control room', 'pinwheel', 'plunge', 'silkworm', 'swamp chestnut oak', 'zinfandel', 'brick red', 'ethyl alcohol', 'hammer', 'black-tailed deer ', 'duckling', 'floss', 'kingfish', 'organ loft', 'vintage', 'gosling']

dataset_folder = "./compressed_dataset/train"

for prompt in prompts :
	prompt_folder = os.path.join(dataset_folder, prompt)
	image_files = os.listdir(prompt_folder)
	i=0
	for img in image_files:
		output_path, images = transform(img,i,prompt)
		image = images[0]
		while is_image_all_black(image):
			output_path, images = transform(img,i)
			image = images[0]
		image.save(output_path)
		i+=1

