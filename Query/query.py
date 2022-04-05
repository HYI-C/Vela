import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from Settings.settings import *

class Query:
    def __init__(
        self,
        embed_univ, # representation of the universe
        data, # the actual universe, name is in index 0, desc is index 1
    ):
        ms = Settings()
        ms.configure()
        self.top_n = ms.top_n
        self.emb_univ = embed_univ
        self.data = data
        self.model = SentenceTransformer(ms.model_name)
        self.sentences = []

    def _similarity(self, emb_query):
        '''This returns the similarity of each item in the universe'''
        similarity = util.pytorch_cos_sim(emb_query, self.emb_univ) 
        print(len(similarity[0]))
        print(len(self.emb_univ))
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
        print(self.top_n)
        sim_set = set(similarity[0])
        for _ in range(0, self.top_n): #we need to put a threshold in here
            #print(max(sim_set))
            if max(sim_set) != 0:
            #print(max(sim_set))
                idx = np.where(similarity[0] == max(sim_set))
                res_inds = np.append(res_inds, idx[0]).astype(int)
                sim_set.remove(max(sim_set))
            else:
                break
        return res_inds
    
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
        res_inds = self._find_top_n_inds(similarity)
        sim_companies = self._return_top_n(res_inds)
        return sim_companies