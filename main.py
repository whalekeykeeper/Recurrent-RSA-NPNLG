# this code will generate a literal caption and a pragmatic caption (referring expression) for the first of the urls provided in the context of the rest

import matplotlib

matplotlib.use("Agg")
import pickle
import re
import time
from collections import defaultdict

import numpy as np
import requests
import tensorflow as tf
from keras.preprocessing import image

from bayesian_agents.joint_rsa import RSA
from recursion_schemes.recursion_schemes import ana_beam, ana_greedy
from utils.config import *
from utils.numpy_functions import make_initial_prior, uniform_vector

#
# urls = [
#     "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Arriva_T6_nearside.JPG/1200px-Arriva_T6_nearside.JPG",
#     "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/First_Student_IC_school_bus_202076.jpg/220px-First_Student_IC_school_bus_202076.jpg"
# ]

# Qin: If an URL is visited too much, wikipedia might ban the visit. A "PIL.UnidentifiedImageError" will occur. In
# this case, just try to get some new URLs.
urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Polar_Bear_AdF.jpg/1599px-Polar_Bear_AdF.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/66/Polar_Bear_-_Alaska_%28cropped%29.jpg",
]
# code is written to be able to jointly infer speaker's rationality and neural model, but for simplicity, let's assume these are fixed
# the rationality of the S1
rat = [100.0]
# the neural model: captions trained on MSCOCO ("coco") are more verbose than VisualGenome ("vg")
model = ["vg"]
number_of_images = len(urls)
# the model starts of assuming it's equally likely any image is the intended referent
initial_image_prior = uniform_vector(number_of_images)
initial_rationality_prior = uniform_vector(1)
initial_speaker_prior = uniform_vector(1)
initial_world_prior = make_initial_prior(
    initial_image_prior, initial_rationality_prior, initial_speaker_prior
)

# make a character level speaker, using torch model (instead of tensorflow model)
speaker_model = RSA(seg_type="char", tf=False)
speaker_model.initialize_speakers(model)
# set the possible images and rationalities
speaker_model.speaker_prior.set_features(images=urls, tf=False, rationalities=rat)
speaker_model.initial_speakers[0].set_features(images=urls, tf=False, rationalities=rat)
# generate a sentence by unfolding stepwise, from the speaker: greedy unrolling used here, not beam search: much better to use beam search generally
literal_caption = ana_greedy(
    speaker_model,
    target=0,
    depth=0,
    speaker_rationality=0,
    speaker=0,
    start_from=list(""),
    initial_world_prior=initial_world_prior,
)

pragmatic_caption = ana_greedy(
    speaker_model,
    target=0,
    depth=1,
    speaker_rationality=0,
    speaker=0,
    start_from=list(""),
    initial_world_prior=initial_world_prior,
)

print("Literal caption:\n", literal_caption)
print("Pragmatic caption:\n", pragmatic_caption)
