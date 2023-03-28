from ...build_test_sets import types
from typing import Union
from utils.numpy_functions import make_initial_prior, uniform_vector
from ...init_models import init_captioner, init_evaluator
from recursion_schemes.recursion_schemes import ana_beam, ana_greedy
from typing import List
from bayesian_agents.rsaState import RSA_State
from bayesian_agents.rsaWorld import RSA_World
from bayesian_agents.joint_rsa import RSA
from utils.config import stop_token
from tqdm import tqdm
import numpy as np


def _get_caption_prob_on_target(evaluator: RSA, caption: str, target: int = 0, n_images: int = 10):
    # no need for SOS becuase it is added automatically,
    # but keep EOS for keeping probability of ending the caption there
    start_from = list(caption.replace("^", ""))
    initial_image_prior = uniform_vector(n_images)
    initial_rationality_prior = uniform_vector(1)
    initial_speaker_prior = uniform_vector(1)
    initial_world_prior = make_initial_prior(
        initial_image_prior, initial_rationality_prior, initial_speaker_prior
    )
    listener_rationality = 1.0
    state = RSA_State(initial_world_prior,
                      listener_rationality=listener_rationality)
    context_sentence = ["^"]
    state.context_sentence = context_sentence
    world = RSA_World(target=target, rationality=0, speaker=0)

    state.timestep = 1
    probs = []
    for char in start_from:
        segment = evaluator.seg2idx[char]
        s = evaluator.speaker(state=state, world=world, depth=1)  # (30,)
        p = s[segment]
        probs.append(p)
        l = evaluator.listener(state=state, utterance=segment, depth=1)
        state.world_priors[state.timestep] = l
        state.context_sentence += [char]
        state.timestep += 1

    summed_probs = np.sum(np.asarray(probs))

    return summed_probs


def _get_evaluator_prediction(urls: List[str], pragmatic_caption: str) -> int:
    evaluator = init_evaluator(urls)

    target_probs = []
    for i in range(len(urls)):
        prob = _get_caption_prob_on_target(
            evaluator, pragmatic_caption, target=i, n_images=len(urls))
        target_probs.append(prob)
    predicted = np.argmax(np.asarray(target_probs))
    return predicted


def evaluate_cluster(cluster: Union[types.TS1_Item, types.TS2_Item]) -> bool:
    urls = [
        cluster["target"]["local_path"],
        *[item["local_path"] for item in cluster["distractors"]]
    ]
    number_of_images = len(urls)
    initial_image_prior = uniform_vector(number_of_images)
    initial_rationality_prior = uniform_vector(1)
    initial_speaker_prior = uniform_vector(1)
    initial_world_prior = make_initial_prior(
        initial_image_prior, initial_rationality_prior, initial_speaker_prior
    )
    captioner = init_captioner(urls)

    pragmatic_caption = ana_beam(
        captioner,
        depth=1,
        beam_width=10,
        initial_word_prior=initial_world_prior,
    )
    print(f"Pragmatic caption beam: {pragmatic_caption[0][0]}")
    listener_prediction = _get_evaluator_prediction(
        urls, pragmatic_caption[0][0])

    if listener_prediction == 0:
        return True
    return False
