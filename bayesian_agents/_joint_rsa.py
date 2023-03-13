import numpy as np
from utils.image_and_text_utils import char_to_index, index_to_char
from train.Model import Model
from typing import List, Union, Tuple
from enum import Enum
from bayesian_agents.rsaWorld import RSA_World
from bayesian_agents.rsaState import RSA_State
from scipy.special import logsumexp


class SamplingStrategy(Enum):
    GREEDY = "greedy"
    BEAM = "beam"


class SpeakerType(Enum):
    LITERAL = "literal"
    PRAGMATIC = "pragmatic"


class RSA:
    def __init__(self, images: List[str] = []) -> None:
        self.idx2seg = index_to_char
        self.seg2idx = char_to_index
        self.n_images = len(images)

        self._speaker_rationality = 0.0

        self.neural_model = Model(
            path="coco", dictionaries=(self.seg2idx, self.idx2seg))
        self.neural_model.set_features(
            images=images, rationalities=[100.0], tf=False)

    def _init_prior(self):
        # start with every image having equal prior
        return np.full((self.n_images, ), 1 / self.n_images)

    def _literal_speaker(
        self,
        partial_caption: List[str] = None,
        target_image_idx: int = None
    ) -> np.ndarray:

        world = RSA_World(target=target_image_idx)
        state = RSA_State(context_sentence=partial_caption)
        return self.neural_model.forward(world, state)

    def _literal_listener(
        self,
        prior: np.ndarray = None,
        utterance: int = None, partial_caption: List[str] = None
    ) -> np.ndarray:

        posterior = np.zeros((self.n_images, ))
        for i in range(self.n_images):
            speaker_posterior = self._literal_speaker(
                partial_caption=partial_caption, target_image_idx=i)
            score = speaker_posterior[utterance]
            posterior[i] = score
        posterior_distribution = (prior + posterior) - \
            logsumexp(prior + posterior)
        return posterior_distribution

    def _pragmatic_speaker(
        self,
        partial_caption: List[str] = None,
        target_image_idx: int = None,
        listener_prior: np.ndarray = None,
    ) -> np.ndarray:

        prior = self._literal_speaker(
            partial_caption=partial_caption, target_image_idx=target_image_idx)

        posterior = np.zeros(prior.shape)

        for k in range(posterior.shape[0]):
            # for every char in vocabulary
            listener_posterior = self._literal_listener(
                prior=listener_prior,
                utterance=k,
                partial_caption=partial_caption,
            )
            posterior[k] = listener_posterior[target_image_idx]

        posterior = self._speaker_rationality * posterior
        return (prior + posterior) - logsumexp(prior + posterior)

    def _sample_greedy(
        self,
        speaker_rationality: float = 0.0,
        target_image_idx: int = 0,
        max_sentence_length: int = 60,
        speaker_type: SpeakerType = SpeakerType.PRAGMATIC,
    ) -> Tuple[str, float]:
        partial_caption = ["^"]
        self._speaker_rationality = speaker_rationality

        probs = []
        next_listener_prior = None

        while len(partial_caption) < max_sentence_length:
            prev_listener_posterior = None
            if next_listener_prior is not None:
                prev_listener_posterior = next_listener_prior
            else:
                prev_listener_posterior = self._init_prior()

            if speaker_type == SpeakerType.LITERAL:
                speaker_posterior = self._literal_speaker(
                    partial_caption=partial_caption,
                    target_image_idx=target_image_idx,
                )
            else:
                speaker_posterior = self._pragmatic_speaker(
                    partial_caption=partial_caption,
                    target_image_idx=target_image_idx,
                    listener_prior=prev_listener_posterior,
                )
            segment = np.argmax(speaker_posterior)
            p = speaker_posterior[segment]
            probs.append(p)

            if speaker_type == SpeakerType.PRAGMATIC:
                prev_listener_posterior = self._literal_listener(
                    prior=prev_listener_posterior,
                    partial_caption=partial_caption,
                    utterance=segment
                )
                next_listener_prior = prev_listener_posterior
            partial_caption.append(self.idx2seg[segment])
            if self.idx2seg[segment] == "$":
                break

        summed_probs = np.sum(np.asarray(probs))
        return ("".join(partial_caption), summed_probs)

    def _sample_beam(
        self,
        speaker_rationality: float = 0.0,
        target_image_idx: int = 0,
        max_sentence_length: int = 60,
        speaker_type: SpeakerType = SpeakerType.PRAGMATIC,
        n_beams: int = 5,
        cut_rate: int = 1,
        max_sentences: int = 50,
    ) -> Tuple[str, float]:
        self._speaker_rationality = speaker_rationality
        initial_caption = ["^"]
        queue: List[Tuple[int, List[str], List[float], np.ndarray]] = []
        final_sentences: List[Tuple[str, float]] = []

        for i in range(n_beams):
            queue.append((i, [*initial_caption], [], self._init_prior()))

        itercount = 0
        while queue:
            if len(final_sentences) >= max_sentences:
                break

            if itercount % (n_beams * cut_rate) == 0:
                # After processing cut_rate timesteps for all beams, we cut the queue
                # and keep only the best sentences
                q = sorted(queue, key=lambda x: np.sum(x[2]), reverse=True)
                queue = q[:n_beams]

            beam_id, partial_caption, sentence_prob, listener_prior = queue.pop(
                0)
            if speaker_type == SpeakerType.LITERAL:
                speaker_posterior = self._literal_speaker(
                    partial_caption=partial_caption,
                    target_image_idx=target_image_idx,
                )
            else:
                speaker_posterior = self._pragmatic_speaker(
                    partial_caption=partial_caption,
                    target_image_idx=target_image_idx,
                    listener_prior=listener_prior,
                )
            segment = np.flip(np.argsort(speaker_posterior))[
                :n_beams][beam_id]

            sentence_prob.append(speaker_posterior[segment])
            char = self.idx2seg[segment]
            partial_caption.append(char)
            if char == "$":
                final_sentences.append(
                    ("".join(partial_caption), np.sum(sentence_prob)))
            elif len(partial_caption) == max_sentence_length:
                final_sentences.append(
                    ("".join(partial_caption), np.sum(sentence_prob)))
            else:
                for i in range(n_beams):
                    if speaker_type == SpeakerType.PRAGMATIC:
                        new_listener_prior = self._literal_listener(
                            prior=listener_prior,
                            partial_caption=partial_caption,
                            utterance=segment
                        )
                    else:
                        new_listener_prior = self._init_prior()  # does not matter
                    queue.append(
                        (i, [*partial_caption], [*sentence_prob], new_listener_prior))
            itercount += 1

        sorted_sentences = sorted(
            final_sentences, key=lambda x: x[1], reverse=True)
        return sorted_sentences[0]

    def sample(self,
               strategy: SamplingStrategy = SamplingStrategy.GREEDY,
               speaker_type: SpeakerType = SpeakerType.PRAGMATIC,
               target_image_idx: int = 0,
               speaker_rationality: float = 5.0,
               n_beams: int = 5,
               cut_rate: int = 2,
               max_sentences: int = 50,
               max_sentence_length: int = 60
               ):
        if strategy == SamplingStrategy.GREEDY:
            return self._sample_greedy(
                speaker_rationality=speaker_rationality,
                target_image_idx=target_image_idx,
                speaker_type=speaker_type,
                max_sentence_length=max_sentence_length,
            )
        elif strategy == SamplingStrategy.BEAM:
            return self._sample_beam(
                cut_rate=cut_rate,
                max_sentence_length=max_sentences,
                max_sentences=max_sentences,
                n_beams=n_beams,
                speaker_rationality=speaker_rationality,
                speaker_type=speaker_type,
                target_image_idx=target_image_idx,
            )

    def compute_posterior(self, caption: str):
        pass
