from pyro.distributions import Dirichlet 
from pyro.distributions import Categorical 

import torch as tr 

from typing import List 

DTYPE = tr.double

class doc_generator: 

    """
    Document Generator based on un-smoothed version of Latent Dirichlet Allocation 
    """

    def __init__(self, M:int, L:int, topic_prior:tr.Tensor, beta = None, eta = None) -> None:

        # topic 1 -> science
        self.topics = \
            {
                0:'science',
                1:'sport',
                2:'art',
                3:'health',
                4:'law',
            }

        self.words = \
            {
                0:{
                    'name':'quantum',
                    'prob':[0.96, 0.01, 0.01, 0.01, 0.01],
                    'topic':'science',
                    },
                1:{
                    'name':'genetics',
                    'prob':[0.50, 0.04, 0.04, 0.40, 0.02],
                    'topic':'science',
                    },
                2:{
                    'name':'research',
                    'prob':[0.7, 0.04, 0.03, 0.2, 0.03],
                    'topic':'science',
                    },
                3:{
                    'name':'scientst',
                    'prob':[0.90, 0.025, 0.025, 0.025, 0.025],
                    'topic':'science',
                   },
                4:{
                    'name':'energy',
                    'prob':[0.4, 0.3, 0.1, 0.1, 0.1],
                    'topic':'science',
                   },
                5:{
                    'name':'electricity',
                    'prob':[0.7, 0.1, 0.05, 0.1, 0.05],
                    'topic':'science',
                    },
                6:{
                    'name':'immunology',
                    'prob':[0.25, 0.2, 0.2,0.25, 0.1],
                    'topic':'science',
                    },
                7:{'name':'astrophysics',
                   'prob':[0.90, 0.025, 0.025, 0.025, 0.025],
                   'topic':'science',
                   },
                8:{
                    'name':'athletics',
                    'prob':[0.05, 0.8, 0.05, 0.05, 0.05],
                    'topic':'sport',
                    },
                9:{
                    'name':'exercise',
                    'prob':[0.04, 0.50, 0.03, 0.40, 0.03],
                    'topic':'sport',
                    },
                10:{
                    'name':'physical',
                    'prob':[0.3, 0.5, 0.15, 0.025, 0.025],
                    'topic':'sport',
                    },
                11:{
                    'name':'football',
                    'prob':[0.05, 0.9, 0.01, 0.01, 0.03],
                    'topic':'sport',
                   },
                12:{
                    'name':'game',
                    'prob':[0.1, 0.4, 0.3, 0.1, 0.1],
                    'topic':'sport',
                   },
                13:{
                    'name':'Olympic',
                    'prob':[0.03, 0.9, 0.03, 0.02, 0.02],
                    'topic':'sport',
                    },
                14:{
                    'name':'recreation',
                    'prob':[0.01, 0.8, 0.1, 0.04, 0.04],
                    'topic':'sport',
                    },
                15:{'name':'FIFA',
                   'prob':[0.01, 0.95, 0.01, 0.01, 0.02],
                   'topic':'sport',
                   },
                16:{
                    'name':'Technique',
                    'prob':[0.1, 0.2, 0.35, 0.2, 0.15],
                    'topic':'art',
                    },
                17:{
                    'name':'content',
                    'prob':[0.1, 0.05, 0.7, 0.05, 0.1],
                    'topic':'art',
                    },
                18:{
                    'name':'Craftsmanship',
                    'prob':[0.2, 0.05, 0.7, 0.025, 0.025],
                    'topic':'art',
                    },
                19:{
                    'name':'form',
                    'prob':[0.2, 0.1, 0.4, 0.1, 0.2],
                    'topic':'art',
                },
                20:{
                    'name':'Symmetrical',
                    'prob':[0.15, 0.15, 0.6, 0.15, 0.15],
                    'topic':'art',
                   },
                21:{
                    'name':'asymmetrical',
                    'prob':[0.15, 0.15, 0.6, 0.15, 0.15],
                    'topic':'art',
                    },
                22:{
                    'name':'picture',
                    'prob':[0.035, 0.035, 0.9, 0.015, 0.015],
                    'topic':'art',
                    },
                23:{'name':'concert',
                   'prob':[0.04, 0.04, 0.9, 0.01, 0.01],
                   'topic':'art',
                   },   
                24:{
                    'name':'allergy',
                    'prob':[0.3, 0.04, 0.04, 0.6, 0.02],
                    'topic':'health',
                    },
                25:{
                    'name':'appetite',
                    'prob':[0.05, 0.2, 0.01, 0.7, 0.04],
                    'topic':'health',
                    },
                26:{
                    'name':'fever',
                    'prob':[0.1, 0.1, 0.1, 0.5, 0.2],
                    'topic':'health',
                    },
                27:{
                    'name':'infection',
                    'prob':[0.3, 0.25, 0.05, 0.35, 0.05],
                    'topic':'health',
                },
                28:{
                    'name':'contagious',
                    'prob':[0.15, 0.05, 0.05, 0.7, 0.05],
                    'topic':'health',
                   },
                29:{
                    'name':'bruise',
                    'prob':[0.10, 0.2, 0.05, 0.6, 0.05],
                    'topic':'health',
                    },
                30:{
                    'name':'decongestant',
                    'prob':[0.025, 0.025, 0.025, 0.9, 0.025],
                    'topic':'health',
                    },
                31:{'name':'injection',
                   'prob':[0.03, 0.03, 0.1, 0.8, 0.04],
                   'topic':'health',
                   },   
                32:{
                    'name':'contract',
                    'prob':[0.025, 0.025, 0.025, 0.025, 0.9],
                    'topic':'law',
                    },
                33:{
                    'name':'bankrupt',
                    'prob':[0.01, 0.01, 0.01, 0.01, 0.96],
                    'topic':'law',
                    },
                34:{
                    'name':'evidence',
                    'prob':[0.3, 0.03, 0.03, 0.04, 0.6],
                    'topic':'law',
                    },
                35:{
                    'name':'court',
                    'prob':[0.025, 0.025, 0.025, 0.025, 0.9],
                    'topic':'law',
                },
                36:{
                    'name':'attorney',
                    'prob':[0.05, 0.04, 0.03, 0.03, 0.85],
                    'topic':'law',
                   },
                37:{
                    'name':'copyright',
                    'prob':[0.1, 0.1, 0.1, 0.1, 0.6],
                    'topic':'law',
                    },
                38:{
                    'name':'accuse',
                    'prob':[0.025, 0.025, 0.025, 0.025, 0.9],
                    'topic':'law',
                    },
                39:{'name':'divorce',
                   'prob':[0.025, 0.025, 0.025, 0.025, 0.9],
                   'topic':'law',
                   },   
            }
        
        # length of the document
        self.L = L
        
        # num topics 
        self.K = 5 
        assert self.K == len(topic_prior)

        # num document 
        self.M = M 

        # number of words (Unique)
        self.V = 40

        self.beta = tr.empty(self.K, self.V, dtype=DTYPE)
        for k,v in self.words.items(): 

            assert v['topic'] in set(self.topics.values()), print(v['topic'])
            assert round(sum(v['prob'])) == 1, print(v['prob'])
            self.beta[:,k] = tr.tensor(v['prob'], dtype=DTYPE) 
        
        for k in range(self.K):
            self.beta[k] = self.beta[k]/self.beta[k].sum()

        assert (tr.round(self.beta.sum(dim=1)) == tr.ones(self.K, dtype=float)).all(), print(self.beta.sum(dim=1),tr.ones(self.K, dtype=float) )
        
        self.alpha = Dirichlet(topic_prior)
        self.theta = self.alpha((self.M,))

    def generate_doc(self, verbose:bool = False,) -> List[List[int]]:

        l = [self.L] * self.M 

        docs = {}
        for d in range(self.M): 

            topic_dist = Categorical(self.theta[d])
            doc = []

            for n in range(l[d]): 

                word_topic = topic_dist()

                word_dist = Categorical(self.beta[word_topic])

                word_n_assignment_idx = word_dist()

                word_n = self.words[word_n_assignment_idx.item()]['name']

                if verbose:
                    print(f"Document: {d} | word: {n} -> topic: {self.topics[word_topic.item()]} -> word: {word_n}")

                doc.append(word_n)
            
            doc_string = " ".join(doc)

            if verbose:
                print(f"Document {d}: {doc_string}")
                print()

            docs[d] = doc_string

        self.docs = docs
        return docs 





