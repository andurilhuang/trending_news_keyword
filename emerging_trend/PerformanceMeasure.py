import watson_developer_cloud
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
  import Features, KeywordsOptions
import nltk
    
def IBM_validation (text_str):
    keywords = []
    natural_language_understanding = NaturalLanguageUnderstandingV1(
  username='139d3c47-11c1-4403-a428-85c3173dd1e5',
  password='IgRYqjyT5CwZ',
  version='2018-03-16')

    response = natural_language_understanding.analyze(
  text=text_str,
  features=Features(
    keywords=KeywordsOptions(
      sentiment=False,
      emotion=False,
      limit=10)), language = 'en')
    
    for keyword in response['keywords']:
        keywords.append(keyword['text'])
    return keywords

def lem_IBM(text):
    import nltk
    lem = nltk.WordNetLemmatizer()
    keywords_new = []
    for word in text:
        constituent = []
        for single in word.split():
            if single[0].isupper() is False:
                constituent.append(lem.lemmatize(single))
            else:
                constituent.append(single)
        keywords_new.append(' '.join(constituent))
    return keywords_new

def evaluate_keywords(proposed,groundtruth):
    # convert to lowercase
    proposed = [word.lower() for word in proposed]
    groundtruth = [word.lower() for word in groundtruth]
    
    proposed_set = set(tuple(proposed))
    true_set = set(tuple(groundtruth))
    
    true_positives = len(proposed_set.intersection(true_set))
    
    if len(proposed_set)==0:
        precision = 0
    else:
        precision = true_positives/float(len(proposed)) 
    
    if len(true_set)==0:
        recall = 0
    else:
        recall = true_positives/float(len(true_set))
    
    if precision + recall > 0:
        f1 = 2*precision*recall/float(precision + recall)
    else:
        f1 = 0
    
    return f1

def evaluate_precision(proposed,groundtruth):
    # convert to lowercase
    proposed = [word.lower() for word in proposed]
    groundtruth = [word.lower() for word in groundtruth]
    
    proposed_set = set(tuple(proposed))
    true_set = set(tuple(groundtruth))
    
    true_positives = len(proposed_set.intersection(true_set))
    
    if len(proposed_set)==0:
        precision = 0
    else:
        precision = true_positives/float(len(proposed)) 
    
    return precision

def evaluate_keywords_subsumed_lower(proposed,groundtruth):
    # convert to lowercase
    proposed = [word.lower() for word in proposed]
    groundtruth = [word.lower() for word in groundtruth]
    
    proposed_set = set(tuple(proposed))
    true_set = set(tuple(groundtruth))
    
    true_positives = len(proposed_set.intersection(true_set))
    
    difference = proposed_set - true_set
    
    for d in difference:
        for t in true_set:
            if len(d.split()) > len(t.split()) and t in d and d.split()[-1] == t.split()[-1]:
                true_positives += 1
    
    if len(proposed_set)==0:
        precision = 0
    else:
        precision = true_positives/float(len(proposed)) 
    
    if len(true_set)==0:
        recall = 0
    else:
        recall = true_positives/float(len(true_set))
    
    if precision + recall > 0:
        f1 = 2*precision*recall/float(precision + recall)
    else:
        f1 = 0
    
    return f1

def evaluate_keywords_subsumed_higher(proposed,groundtruth):
    # convert to lowercase
    proposed = [word.lower() for word in proposed]
    groundtruth = [word.lower() for word in groundtruth]
    
    proposed_set = set(tuple(proposed))
    true_set = set(tuple(groundtruth))
    
    true_positives = len(proposed_set.intersection(true_set))
    
    difference = proposed_set - true_set
    
    for d in difference:
        for t in true_set:
            if len(d.split()) > len(t.split()) and t in d:
                true_positives += 1
    
    if len(proposed_set)==0:
        precision = 0
    else:
        precision = true_positives/float(len(proposed)) 
    
    if len(true_set)==0:
        recall = 0
    else:
        recall = true_positives/float(len(true_set))
    
    if precision + recall > 0:
        f1 = 2*precision*recall/float(precision + recall)
    else:
        f1 = 0
    
    return f1