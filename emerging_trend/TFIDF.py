from itertools import islice
from textblob import TextBlob as tb
import collections, math

def tf(word, blob):
    import collections, math
    #return blob.words.count(word) / len(blob.words)
    d = collections.Counter(blob)
    return d[word] / len(blob)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)



def runmytfidf(data, method):
    import collections, math
    TechnicalTerm = data[method].tolist()
        
    returnList=[]
    #Obtaing the Top Key words for all the pages
    for i, blob in enumerate(TechnicalTerm):
        scores = {word: tfidf(word, blob, TechnicalTerm) for word in blob}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        topWords=[]
        for word, score in sorted_words[:10]:
            topWords.append(word)
        returnList.append(topWords)

    return returnList

from itertools import chain, groupby
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
from operator import itemgetter

def keyphrase_extraction_tfidf(data, method, min_df=5, max_df=0.8, num_key=10):
   
    texts = data['text'].tolist()
    if method == 'NP':
        vocabulary = [keyword_candidate_nounphrase(text) for text in texts]
    elif method == 'TT':
        vocabulary = [keyword_candidate_technicalterm(text) for text in texts]
    elif method == 'SW':
        vocabulary = [keyword_candidate_singleword(text) for text in texts]
    vocabulary = list(chain(*vocabulary))
    vocabulary = list(np.unique(vocabulary)) # unique vocab

    max_vocab_len = max(map(lambda s: len(s.split(' ')), vocabulary))
    tfidf_model = TfidfVectorizer(vocabulary=vocabulary, lowercase=True,
                                  ngram_range=(1,max_vocab_len), stop_words=None,
                                  min_df=min_df, max_df=max_df)
    X = tfidf_model.fit_transform(texts)
    vocabulary_sort = [v[0] for v in sorted(tfidf_model.vocabulary_.items(),
                                            key=operator.itemgetter(1))]
    sorted_array = np.fliplr(np.argsort(X.toarray()))

    # return list of top candidate phrase
    key_phrases = list()
    for sorted_array_doc in sorted_array:
        key_phrase = [vocabulary_sort[e] for e in sorted_array_doc[0:num_key]]
        key_phrases.append(key_phrase)

    return key_phrases