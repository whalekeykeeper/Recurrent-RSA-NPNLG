from typing import List

from utils.numpy_functions import make_initial_prior, uniform_vector
from bayesian_agents.joint_rsa import RSA


def init_evaluator(urls: List[str]):
    rat = [100.0]
    # the neural model: captions trained on MSCOCO ("coco") are more verbose than VisualGenome ("vg")
    model = ["coco"]
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
    speaker_model.speaker_prior.set_features(
        images=urls, tf=False, rationalities=rat)
    speaker_model.initial_speakers[0].set_features(
        images=urls, tf=False, rationalities=rat)

    return speaker_model
