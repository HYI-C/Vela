import pandas as pd
from sklearn.model_selection import train_test_split
from Settings.settings import *

class ConfigData:
    '''This module takes the data in as a csv file and returns a list of
    companies and their descriptions. It removes useless words as defined in the
    settings file, and it also trims the descriptions to a set number of words.
    
    This module can be called in 3 different ways:
    1. if we call module.run(), we configure the entire company universe
    2. if we call module.run_IT(), we configure only IT companies
    3. if we call module.run_test(), we configure the first 10k rows
    
    Example use: 
    data = ConfigData().run()'''
    def __init__(
        self,
    ):
        ms = Settings()
        ms.configure()
        self.char_max = ms.char_max
        self.word_max = ms.word_max
        self.data_path = ms.data_path
        self.ignore_words = ms.ignore_words
        self.data = []

    def _trim_description(self, desc):
        #This function takes in a sentence and trims it to fit our model
        words = desc.split()
        temp = [word for word in words if word not in self.ignore_words]
        if len(temp) < self.word_max:
            return " ".join(temp)
        else:
            return " ".join(temp[0:self.word_max])
        '''   The below method puts limits on characters instead
        try:
            if len(desc) < self.char_max:
                return desc
            else:
                return desc[0:self.char_max]
        except:
            return desc'''
        
    def _build_dict(self, name, desc):
        #This function builds a dictionary of companies and the respective descriptions
        self.data.append([name, desc])

    def run(self):
        df = pd.read_csv(self.data_path)
        df = df.drop(["uuid", "uuid_1"], axis=1)
        for i in range(0, len(df)): #len(df)
            desc = str(df[i:i+1]['description'].values[0])
            name = df[i:i+1]['name'].values[0]
            trimmed_desc = self._trim_description(desc)
            self._build_dict(name, trimmed_desc)
        return self.data

    def run_IT(self):
        '''Here, we only select the subgroup of IT companies, on short desc.'''
        df = pd.read_csv(self.data_path)
        df = df.drop(["uuid", "uuid_1"], axis=1)
        for i in range(0, len(df)):
            IT = str(df[i:i+1]["category_groups_list"].values[0]).split()
            if "Information" in IT:
                desc = str(df[i:i+1]['short_description'].values[0])
                name = df[i:i+1]['name'].values[0]
                trimmed_desc = self._trim_description(desc)
                self._build_dict(name, trimmed_desc)
        return self.data

    def run_test(self):
        df = pd.read_csv(self.data_path)
        df = df.drop(["uuid", "uuid_1"], axis=1)
        for i in range(0, 10000): #we run on the first 10k rows
            desc = str(df[i:i+1]['description'].values[0])
            name = df[i:i+1]['name'].values[0]
            trimmed_desc = self._trim_description(desc)
            self._build_dict(name, trimmed_desc)
        return self.data