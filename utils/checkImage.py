"""
To check if PIL can visit to a certain iamge url or not.
"""
import PIL
from pathlib import Path
from PIL import UnidentifiedImageError
p = 'data/google_images/imghttps:upload.wikimedia.orgwikipediacommonsthumbdd9First_Student_IC_school_bus_202076' \
  '.jpg220px-First_Student_IC_school_bus_202076.jpg.jpg'
import PIL
from PIL import UnidentifiedImageError
import glob

imgs_ = glob.glob(p)
for img in imgs_:
    try:
        img = PIL.Image.open(img)
    except PIL.UnidentifiedImageError:
        print(img)
        print(img._exclusive_fp)