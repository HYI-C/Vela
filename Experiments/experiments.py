from Settings.settings import *
from Data.config_data import *
from Model.model import *
from Query.query import *

ms = Settings()
ms.configure()

import Settings

class Experiment:
    def __init__(
        self, 
    ):
        self.word_max = ms.word_max

    def run(self):
        i = 10
        num_words = []
        score = []
        while i < 31:
            
            Settings.custom.word_max = i
            ms.configure()
            data = ConfigData().run_test()

            model = Model(data)
            embed_univ = model.run()

            query = Query(embed_univ, data)
            sco = query.experiment("Klink Mobile, Inc. is a mobile payments company that provides a secure global application of wireless infrastructure that enables people from around the globe to transfer money and exchange value instantly. While solving one of the biggest payments challenges of this decade, Klink Mobile has developed an all-encompassing mobile wallet solution that positively and directly impacts bottom-up economies by allowing customers to pay bills, purchase airtime or product, and provide international remittance and money transfers worldwide from the comfort and safety of their very own mobile devices.  Klink Mobile’s proprietary cloud-based platform is compatible with all technologies and is account, network, and third-party agnostic enabling interoperability with organization totality within the mobile payments ecosystem. Led by a mission driven team, Klink Mobile is delivering efficient, reliable, technically savvy, and affordable financial tools directly to those who need it most.")
            
            num_words.append(i)
            score.append(sco)
            i+=5
        return num_words, score