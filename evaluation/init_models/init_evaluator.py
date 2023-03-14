from typing import List

from utils.numpy_functions import make_initial_prior, uniform_vector
from bayesian_agents._joint_rsa import RSA


def init_evaluator(urls: List[str]):
    speaker_model = RSA(images=urls, urls_are_local=True, model="vg")

    return speaker_model
