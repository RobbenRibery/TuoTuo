from pyro.distributions import Dirichlet 
from pyro.distributions import Categorical 

import torch as tr 

from typing import List 

class doc_generator: 

    """
    Document Generator based on un-smoothed version of Latent Dirichlet Allocation 
    """

    def __init__(self, M:int, L:int, topic_prior:tr.Tensor, beta = None, eta = None) -> None:

        # topic 1 -> science
        self.topics = [
            {
                0:'science',
                1:'sport',
                2:'art',
                3:'health',
                4:'law',
            }
        ]
        self.topics_words = \
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
            }
        #     'sport':[
        #         {
        #             8:'athletics',
        #             9:'exercise',
        #             10:'physical',
        #             11:'football',
        #             12:'game',
        #             13:'Olympic',
        #             14:'recreation',
        #             15:'FIFA',
        #         }
        #     ],
        #     'art':[
        #         {
        #             16:'Technique',
        #             17:'Craftsmanship',
        #             18:'form',
        #             19:'content',
        #             20:'Symmetrical',
        #             21:'asymmetrical',
        #             22:'picture',
        #             23:'portrait',
        #         }
        #     ],
        #     'health':[
        #         {
        #             24:'Technique',
        #             25:'Craftsmanship',
        #             26:'form',
        #             27:'content',
        #             28:'Symmetrical',
        #             29:'asymmetrical',
        #             30:'picture',
        #             31:'portrait',
        #         }
        #     ]
        # }

#         ache
# allergy
# antihistamine
# appetite
# aspirin
# bandage
# blood
# bone
# broken
# bronchitis
# bruise
# cast
# clinic
# cold
# contagious
# cough
# crutch
# cut
# decongestant
# diarrhea
# dizzy
# fever
# first aid
# flu
# headache
# hives
# indigestion
# infection
# influenza
# injection

        self.topics

        # length of the document
        self.L = L
        
        # num topics 
        self.K = 5 

        # num document 
        self.M = M 

        # number of words (Unique)
        self.V = 40
        
        self.alpha = Dirichlet(topic_prior)
        self.theta = self.alpha(tr.Size(self.M, self.K))

    def generate_doc(self,) -> List[List[int]]:

        docs = []
        for d in range(self.theta.shape[0]): 

            topic_dist = Categorical(self.theta[d])
            doc = []

            for n in range(self.L): 

                word_topic = topic_dist(tr.Size(1,))

                word_dist = Categorical(self.beta[word_topic])

                word_n_assignment_idx = word_dist(tr.Size(1,))

                word_n = self.vocab[word_n_assignment_idx]

                doc.append(word_n)

            docs.append(doc)

        self.docs = doc





