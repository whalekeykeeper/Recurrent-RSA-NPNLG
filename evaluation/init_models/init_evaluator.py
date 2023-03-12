from typing import List

from utils.numpy_functions import make_initial_prior, uniform_vector
from bayesian_agents.joint_rsa import RSA


def init_evaluator(urls: List[str]):
    rat = [100.0]
    # the neural model: captions trained on MSCOCO ("coco") are more verbose than VisualGenome ("vg")
    model = ["vg"]

    # make a character level speaker, using torch model (instead of tensorflow model)
    speaker_model = RSA(seg_type="char", tf=False)
    speaker_model.initialize_speakers(model)
    # set the possible images and rationalities
    speaker_model.speaker_prior.set_features(
        images=urls, tf=False, rationalities=rat, urls_are_local=True)
    speaker_model.initial_speakers[0].set_features(
        images=urls, tf=False, rationalities=rat, urls_are_local=True)

    return speaker_model
