import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from Settings.settings import *

class Query:
    def __init__(
        self,
        embed_univ, # representation of the universe
        data, # the actual universe, name is in index 0, desc is index 1
        evaluate = True,
        model = False,
    ):
        ms = Settings()
        ms.configure()
        self.top_n = ms.top_n
        self.eval_threshold = ms.eval_threshold
        self.sentences1 = ms.sentences1
        self.sentences2 = ms.sentences2
        self.scores = ms.scores
        self.emb_univ = embed_univ
        self.data = data
        self.evaluate = evaluate
        self.model = SentenceTransformer(ms.model_name)
        self.sentences = []
        if model:
            self.model = model

    def _similarity(self, emb_query):
        '''This returns the similarity of each item in the universe'''
        similarity = util.pytorch_cos_sim(emb_query, self.emb_univ) 
        return similarity
    
    def _embed(self, item):
        '''We embed the query item into representation space'''
        emb_query = self.model.encode(item, convert_to_tensor=True)
        #print(item)
        return emb_query
    
    def _find_top_n_inds(self, similarity): 
        '''this function takes in the image data and returns the bins along with
        the index of the bin where each point belongs'''
        res_inds = np.array([])
        max_scores = np.array([])
        sim_set = set(similarity[0])
        for _ in range(0, self.top_n): #we need to put a threshold in here
            if max(sim_set) != 0:
                idx = np.where(similarity[0] == max(sim_set))
                res_inds = np.append(res_inds, idx[0]).astype(int)
                max_scores = np.append(max_scores, max(sim_set))
                sim_set.remove(max(sim_set))
            else:
                break
        return res_inds, max_scores
    
    def _return_top_n(self, res_inds):
        '''Here, we return the top n companies'''
        sim_companies = []
        for idx in res_inds:
            sim_companies.append(self.data[idx])
        #sim_companies = self.data[map_]
        return sim_companies

    def run(self, description):
        emb_query = self._embed(description)
        similarity = self._similarity(emb_query)
        res_inds, max_scores = self._find_top_n_inds(similarity)
        sim_companies = self._return_top_n(res_inds)
        if self.evaluate:
            #Find the number of "good scores"
            check = np.where(max_scores > self.eval_threshold)
            num_good = len(check[0])
            #This is to check against different sentences
            evaluator = EmbeddingSimilarityEvaluator(self.sentences1, self.sentences2, self.scores, write_csv= True)
            score = evaluator(self.model)
            return sim_companies, num_good, score
        else:
            return sim_companies

    def experiment(self, description):
        emb_query = self._embed(description)
        similarity = self._similarity(emb_query)
        _, max_scores = self._find_top_n_inds(similarity)
        return sum(max_scores)
