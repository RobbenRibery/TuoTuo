import re
import unicodedata
import string
import spacy
import nltk
#nltk.download()
from nltk.tokenize import ToktokTokenizer
from urllib.request import Request, urlopen

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
#eng_words = set(nltk.corpus.words.words())
# custom: removing words from list
stopword_list.remove('not')
import nltk
words = set(nltk.corpus.words.words())

# function to extract the text according to source 
#def get_text(url,source):  


# function to remove accented characters  
def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text 

# function to remove special characters
def remove_special_characters(text):
    # define the pattern to keep
    pat = r"[^a-zA-z0-9.,!?/:;\"\'\s]"
    return re.sub(pat, ' ', text)

# function to remove numbers
def remove_numbers(text):
    # define the pattern to keep
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    return re.sub(pattern, '', text)

# function to remove punctuation
def remove_punctuation(text):
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

# function for stemming
#def get_stem(text):
    #stemmer = nltk.porter.PorterStemmer()
    #text = ' '.join([stemmer.stem(word) for word in text.split()])
    #return text

# def get_lem(text):
#     text = nlp(text)
#     text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
#     return text

# function to remove stopwords
def remove_stopwords(text):
    # convert sentence into token of words
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    # check in lowercase 
    t = [token for token in tokens if token.lower() not in stopword_list]
    text = ' '.join(t)    
    return text

# function to remove special characters
def remove_extra_whitespace_tabs(text):
    #pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()


def text_pipline(text, forbidden_list): 

    t1 = remove_accented_chars(text)
    t1 = remove_special_characters(t1)
    t1 = remove_numbers(t1)
    t1 = remove_punctuation(t1)

    #t1_6 = get_lem(t1_4)
    t1 = remove_stopwords(t1) 
    t1 = remove_extra_whitespace_tabs(t1)
  
    temp_list = [x for x in t1.split(' ') if x not in forbidden_list and 'http' not in x]

    if len(temp_list) <= 4: 
        return 'NOINFO'
    else: 
        t1_8 = " ".join(temp_list)
        t2 = " ".join(w for w in nltk.wordpunct_tokenize(t1_8) if w.lower() in words or not w.isalpha())

    return t2.lower()

def list_generator(x,forbidden_list=None): 
    """
    here x intake is string 
    """
    list1 = x.replace("[",'').replace("]",'').replace("'",'').split(',')
    list2 = []
    for i in range(len(list1)): 

        if 'http' in list1[i]: 
            pass 
        if list1[i] in forbidden_list: 
            pass 
        else: 
            if i == 0 and list1[i]: 
                list2.append(list1[i])
            elif i != 0 and list1[i]:
                list2.append(list1[i][1:])
            else: 
                pass
    
    list2 = [x for x  in list2 if x not in forbidden_list and 'http' not in x]
    
    #" ".join(w for w in nltk.wordpunct_tokenize(sent) if w in list2 or not w.isalpha())

    return list2