import pandas as pd
from sklearn.model_selection import train_test_split
from Settings.settings import *

class ConfigData:
    '''This creates a list of companies and their descriptions in a clean
    format'''
    def __init__(
        self,
    ):
        ms = Settings()
        ms.configure()
        self.char_max = ms.char_max
        self.data_path = ms.data_path
        self.data = []

    def _trim_description(self, desc):
        '''This function takes in a sentence and trims it to fit our model'''
        try:
            if len(desc) < self.char_max:
                return desc
            else:
                return desc[0:self.char_max]
        except:
            return desc
        
    def _build_dict(self, name, desc):
        '''This function builds a dictionary of companies and the respective
        descriptions'''
        self.data.append([name, desc])

    def run(self):
        df = pd.read_csv(self.data_path)
        df = df = df.drop(["uuid", "uuid_1"], axis=1)
        for i in range(0, len(df)): #len(df)
            desc = df[i:i+1]['description'].values[0].astype(str)
            name = df[i:i+1]['name'].values[0]
            trimmed_desc = self._trim_description(desc)
            self._build_dict(name, trimmed_desc)
        return self.data
