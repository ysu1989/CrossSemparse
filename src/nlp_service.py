import string
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

# remove punctuation?
REMOVE_PUNCT = False
# lower case?
LOWER_CASE = True
# lemmatize?
LEMMATIZE = False

def tokenize(utter):
    """tokenize an utterance"""
    if LEMMATIZE:
        annotators = 'tokenize, ssplit, lemma'
    else:
        annotators = 'tokenize, ssplit'
    out_format = 'json'
    res = nlp.annotate(utter, properties={'annotators': annotators,
                                          'outputFormat': out_format})
    tokens = []
    for s in res['sentences']:
        if LEMMATIZE:
            tokens += [token['lemma'] for token in s['tokens']]
        else:
            tokens += [token['word'] for token in s['tokens']]

    if LOWER_CASE:
        tokens = [token.lower() for token in tokens]

    if REMOVE_PUNCT:
        tokens = remove_punct(tokens)

    return tokens

def remove_punct(tokens):
    """remove punctuations (defined by string.punctuation)"""
    exclude = set(string.punctuation)
    tokens = [token for token in tokens if token not in exclude]
    return tokens
