# this code will generate a literal caption and a pragmatic caption (referring expression) for the first of the urls provided in the context of the rest

from collections import defaultdict

import matplotlib
from keras.preprocessing import image

from bayesian_agents._joint_rsa import RSA, SamplingStrategy, SpeakerType
from utils.config import *

matplotlib.use("Agg")

urls = [
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRnaWSDM__KaWGxTPrpMpwF5DvgJ-U4cfyL4g&usqp=CAU",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcrNMOpyvmHPShc9fb6tLar5UJnReaNmj7bQ&usqp=CAU",
]


# the model starts of assuming it's equally likely any image is the intended referent
model = RSA(images=urls)

GREEDY_caption = model.sample(
    strategy=SamplingStrategy.GREEDY,
    speaker_type=SpeakerType.LITERAL,
    target_image_idx=1,
    speaker_rationality=5
)

BEAM_caption = model.sample(
    strategy=SamplingStrategy.BEAM,
    speaker_type=SpeakerType.PRAGMATIC,
    target_image_idx=1,
    speaker_rationality=5,
    n_beams=10,
    cut_rate=1
)

print("GREEDY caption:\n", GREEDY_caption)
print("BEAM caption:\n", BEAM_caption)
