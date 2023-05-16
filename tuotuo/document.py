from typing import List, Dict, Union  

import pandas as pd 

from tuotuo.utils import (
    process_documents
)

class Document: 

    """
    Class that performs preprocssing on input documents
    Collect statistics about the document and thus would be
    used in the LDA module 

    """

    def __init__(
        self, 
        sample_docs: dict,
        num_doc_population:int = None,
    ) -> None:
        
        self.sample_docs = sample_docs

        if not num_doc_population: 
            self.num_doc_population = len(sample_docs)
        else:
            self.num_doc_population = num_doc_population 

        preproc_result = process_documents(self.sample_docs, sample=False,)

        self.sample_docs_list = preproc_result['documents']
        self.vocab_doc_count_dict = preproc_result['vocab_doc_count_dict']
        self.vocab_doc_count_array = preproc_result['vocab_doc_count_array']
        self.doc_vocab_count_array = self.vocab_doc_count_array.T 

        self.vocab_to_idx = preproc_result['vocab_to_idx'] 
        self.idx_to_vocab = preproc_result['idx_to_vocab']

        self.doc_vocab_count_df = pd.DataFrame(
            data = self.doc_vocab_count_array,
            columns = list(self.vocab_to_idx.keys())
        )

        # Number of unique vocab
        self.V = self.vocab_doc_count_array.shape[0]

        # number of documments
        self.M = self.vocab_doc_count_array.shape[1]



