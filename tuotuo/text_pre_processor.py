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

if __name__ == "__main__":
    
    ex = '\\n\n\nText size\n\n\n\n\n\n\n\n\n\nShares of online local listings and review site Yelp (YELP) are down $4.61, or 11%, at $37.83, following a story this afternoon by Bloomberg\'s Alex Sherman stating that the company has decided not to pursue a sale of itself despite hiring Goldman Sachs to prospect for suitors, and actually finding "several."\nThe article, citing multiple unnamed sources, states that it\'s up to CEO Jeremy Stoppelman to "change his mind" if the site is to resume talks.\n First to comment this afternoon from the Street is Victor Anthony of Axiom Capital, who argues Yelp "did not find a buyer," and that actually Google (GOOGL) and Facebook (FB) and Amazon.com (AMZN) are replicating the business.\nAnthony, who has a Hold rating on Yelp stock, and a $43 price target, writes that there really aren\'t many credible suitor prospects:\n\nWe argued several times that the list of potential acquirers of Yelp touted by investors such as Google, Facebook, Amazon, and Yahoo! (YHOO) were unlikely to purchase Yelp. Plus, Barry Diller of IAC/InterActiveCorp (IACI) stated publicly that he is not interested in purchasing Yelp, leaving few credible options.\nHe notes "Google is pushing into local businesses and encouraging them to have their customers leave reviews on Google  a direct competitive threat to Yelp. Further, Google Maps has been bolstered by the acquisition of Zagat."\nAs for Facebook, "Many of the same restaurants we see on Yelp are also rated on Facebook," adding "We are seeing signs that Facebook is moving in the direction of allowing a few key Yelp-like features and functionality, with the addition of the \'Book a Table Online\' feature we saw on the Palm Restaurant, which is powered by OpenTable (OPEN)."\nYelp could be a good asset for Yahoo!, he writes, but "we do not see how Yelp fits into Yahoos stated MaVeNS investment growth strategy," and moreover, the potential price tag of $4B might be a non-starter for Yahoo!\n\n'
    print(text_pipline(ex))