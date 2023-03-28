# Qin: This script seems to be used nowhere.

import numpy as np

from utils.image_and_text_utils import max_sentence_length, vectorize_caption
from typing import List


class RSA_State:
    def __init__(self, context_sentence: List[str]):
        self.context_sentence = context_sentence
