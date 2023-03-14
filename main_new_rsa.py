# this code will generate a literal caption and a pragmatic caption (referring expression) for the first of the urls provided in the context of the rest

from collections import defaultdict

import matplotlib
from keras.preprocessing import image

from bayesian_agents._joint_rsa import RSA, SamplingStrategy, SpeakerType
from utils.config import *
import os

matplotlib.use("Agg")

urls = [
]

data_dir = "data/test_sets/ts1/man/3"
for ip in os.listdir(data_dir):
    urls.append(os.path.join(data_dir, ip))


# the model starts of assuming it's equally likely any image is the intended referent
model = RSA(images=urls, urls_are_local=True)

GREEDY_caption = model.sample(
    strategy=SamplingStrategy.GREEDY,
    speaker_type=SpeakerType.PRAGMATIC,
    target_image_idx=5,
    speaker_rationality=5,
    n_beams=10,
    cut_rate=1
)

BEAM_caption = model.sample(
    strategy=SamplingStrategy.BEAM,
    speaker_type=SpeakerType.PRAGMATIC,
    target_image_idx=5,
    speaker_rationality=5,
    n_beams=10,
    cut_rate=1
)

print("GREEDY caption:\n", GREEDY_caption, len(GREEDY_caption[0]))
print("BEAM caption:\n", BEAM_caption, len(BEAM_caption[0]))
