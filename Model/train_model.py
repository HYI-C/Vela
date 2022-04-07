import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import math
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from transformers import InputExample
from Settings.settings import *

class TrainModel:
    def __init__(
        self,
        
    ):
        ms = Settings()
        ms.configure()

    def _configure_model(self):
        '''The architecture of this model is a pooling layer on top of the fully connected layer 
        for dimension reduction.'''

        self.emb_model = models.Transformer(ms.transfomer_name, max_seq_length=ms.char_max)
        self.pooling_model = models.Pooling(self.emb_model.get_word_embedding_dimension())
        self.dense_model = models.Dense(in_features=self.pooling_model.get_sentence_embedding_dimension(), out_features = ms.char_max, activation_function = ms.activation_func)
        
        self.model = SentenceTransformer(modules=[self.emb_model, self.pooling_model, self.dense_model])
        return
    
    def _configure_data(self):
        train_examples = [InputExample(texts=["sentence1", "sentence2"], label=0.8), InputExample(texts=["unlike sentence 1", "unlike sentence 2"], label=0.2)]
        train_dataloader = DataLoader(train_examples, batch_size = ms.batch_size)
        train_loss = losses.CosineSimilarityLoss(self.model)
    
    def train(self, epochs):
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs = epochs)