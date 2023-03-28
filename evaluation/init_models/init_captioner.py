from typing import List

from utils.numpy_functions import make_initial_prior, uniform_vector
from bayesian_agents._joint_rsa import RSA


def init_captioner(urls: List[str]) -> RSA:
    # the neural model: captions trained on MSCOCO ("coco") are more verbose than VisualGenome ("vg")

    # make a character level speaker, using torch model (instead of tensorflow model)
    speaker_model = RSA(images=urls, urls_are_local=True, model="coco")

    return speaker_model
