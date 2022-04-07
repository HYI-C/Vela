import torch
import numpy as np
import math
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from Settings.settings import *

class Model:
    def __init__(
        self,
        data,
    ):
        ms = Settings()
        self.data = data
        self.model = SentenceTransformer(ms.model_name)
        self.sentences = []

    def _construct_sentences(self):
        for i in range(0, len(self.data)):
            self.sentences.append(self.data[i][1])
        return 

    def _construct_embedding(self, all_sentences):
        embeddings = self.model.encode(all_sentences, convert_to_tensor=True)
        return embeddings

    def run(self):
        self._construct_sentences()
        embed_univ = self._construct_embedding(self.sentences) #this is our entire dictionary embedded
        return embed_univ
