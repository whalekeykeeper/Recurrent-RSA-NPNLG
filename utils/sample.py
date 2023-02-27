import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os

from torch.autograd import Variable
from torchvision import transforms
from utils.build_vocab import Vocabulary
from PIL import Image
import re


def to_var(x):
    with torch.no_grad():
        if torch.cuda.is_available():
            x = x.cuda()
    return x


def load_image_from_path(path, transform=None):
    from PIL import Image as PIL_Image

    # Qin
    with Image.open(path) as image:
    # image = Image.open(path)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        # image = image.crop([0,0,224,224])
        if transform is not None:
            image = transform(image).unsqueeze(0)

    return image


def load_image(url, transform=None):
    import urllib.request
    from PIL import Image as PIL_Image
    import shutil
    import requests
    hashed_url = re.sub('/', '', url)
    response = requests.get(url, stream=True)
    with open('data/google_images/img' + hashed_url + '.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    # del response

    if "200" not in str(response):
        print("Check if you are forbbiden to visit the urls.")

    # print(os.listdir())

    image = Image.open('data/google_images/img' + hashed_url + '.jpg')
    # print("image loaded (sample.py)")
    image = image.resize([224, 224], Image.Resampling.LANCZOS)
    # width = image.size[0]
    # height = image.size[1]

    # if width>height:
    #     new_height=224
    #     new_width=224 * (width/height)
    # else:
    #     new_width=224
    #     new_height=224 * (height/width)

    # b = image.resize([int(new_width),int(new_height)],PIL_Image.LANCZOS)
    # # b = a.thumbnail([224, 224], PIL_Image.LANCZOS)
    # # c = b.crop([0,0,224,224])
    # image = b.crop([0,0,224,224])

    if transform is not None:
        image = transform(image).unsqueeze(0)

    # image = transforms.ToTensor()(image).unsqueeze(0)

    return image
