import torch
import numpy as np
import math
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from Settings.settings import *

class Model:
    '''This is the model module for using the pre-trained model. This model
    takes data from the config_data module and uses SBERT to embed the entire
    company universe into 768 dimensional space. This model can also be
    configured to use a custom embedder.
    
    Inputs: data from config_date, optionally: custom embedding model
    Outputs: embedded universe, a 768xn matrix, where n is the length of the
    dataset.
    
    Example use: 
    Stock model:
    embed_univ = Model(data).run()

    Custom model:
    embed_univ = Model(data, model = [NAME OF CUSTOM MODEL]).run()
    '''
    def __init__(
        self,
        data,
        model = None,
    ):
        ms = Settings()
        self.data = data
        self.model = SentenceTransformer(ms.model_name)
        self.sentences = []
        if model:
            self.model = model 

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
