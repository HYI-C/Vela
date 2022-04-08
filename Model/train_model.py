import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import math
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from Settings.settings import *

class TrainModel:
    def __init__(
        self,    
        pairs,    
    ):
        ms = Settings()
        ms.configure()
        self.sentences1 = ms.sentences1
        self.sentences2 = ms.sentences2
        self.scores = ms.scores
        self.pairs = pairs

    def _configure_model(self):
        '''The architecture of this model is a pooling layer on top of the fully connected layer 
        for dimension reduction.'''
        ms = Settings()
        self.emb_model = models.Transformer(ms.transformer_name, max_seq_length=ms.char_max)
        self.pooling_model = models.Pooling(self.emb_model.get_word_embedding_dimension())
        self.dense_model = models.Dense(in_features=self.pooling_model.get_sentence_embedding_dimension(), out_features = ms.char_max, activation_function = ms.activation_func)
        
        self.model = SentenceTransformer(modules=[self.emb_model, self.pooling_model, self.dense_model])
        return
    
    def _input_data(self, pairs):
        train_examples = []
        for i in range(0, len(pairs)):
            input_ = InputExample(texts = [pairs[i][1], pairs[i][3]], label = 0.99)
            #InputExample(texts=["sentence1", "sentence2"], label=0.8)
            train_examples.append(input_)
        return train_examples
        # train_examples = [InputExample(texts=["sentence1", "sentence2"], label=0.8), InputExample(texts=["unlike sentence 1", "unlike sentence 2"], label=0.2)]

    def _configure_data(self, train_examples):
        ms = Settings()
        train_dataloader = DataLoader(train_examples, batch_size = ms.batch_size)
        train_loss = losses.CosineSimilarityLoss(self.model)
        return train_dataloader, train_loss
        
    def train(self, epochs):
        self._configure_model()
        train_examples = self._input_data(self.pairs)
        train_dataloader, train_loss = self._configure_data(train_examples)

        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs = epochs)
        
        evaluator = EmbeddingSimilarityEvaluator(self.sentences1, self.sentences2, self.scores, write_csv= True)
        score = evaluator(self.model)
        print(score)
        return