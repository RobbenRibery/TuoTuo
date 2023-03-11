import pytest 

from src.utils import get_vocab_from_docs 


@pytest.fixture
def input_(): 

    docs = [
        ['My name is tuo tuo .'],
        ['tuo tuo is cute !'],
        ['Tuo tuo likes to poo a lot !'],
    ]

    docs = [doc[0].split(' ') for doc in docs]

    return docs  


def test_vocab_dict(input_): 


    assert get_vocab_from_docs(input_) == \
        {
            'My': [1, 0, 0],
            'name': [1, 0, 0],
            'is': [1, 1, 0],
            'tuo': [2, 2, 1],
            '.': [1, 0, 0],
            'cute': [0, 1, 0],
            '!': [0, 1, 1],
            'Tuo': [0, 0, 1],
            'likes': [0, 0, 1],
            'to': [0, 0, 1],
            'poo': [0, 0, 1],
            'a': [0, 0, 1],
            'lot': [0, 0, 1]
    }