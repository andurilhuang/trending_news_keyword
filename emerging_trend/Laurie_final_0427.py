'''
- Laurie's part:
    -> Text Preprocessing
    -> Noun Phrase Extraction
    -> Keyword Extraction(tf-idf)
    -> Keyword Postprocessing
    -> Verb Phrase Extraction

- The last function 'overall' goes through the whole pipeline in my part: from raw text to processed keyword 
  ready to be used in Su's trending keyword extraction

- Note that the first function is only for Anna's reference, not including in the pipeline

- Date: 04/26/2018

'''

def Anna_text_edit(text):
    '''
    In regards to what we've discussed in the meeting, consider case without : sign 
    also keep only certain punctuations to ensure maximum amount of special characters are removed

    '''
    text = re.sub(r'http\S+', '',text, flags = re.IGNORECASE)
    text = re.sub(r'www.\S+', '',text, flags = re.IGNORECASE)
    # remove all special characters apart from &/.–,’';:?!
    #if this one is used please remove the same line in the data_processing function
    text = re.sub("[^a-zA-Z0-9-&/.–,’';:?!\s]", '', text)
    # add * after : sign
    text = re.sub(r'\bmobile\s*:*\s*\b', '',text, flags = re.IGNORECASE)
    text = re.sub(r'\bphone\s*:*\s*\b', '',text, flags = re.IGNORECASE)
    text = re.sub(r'\be*-*mail\s*:*\s*\b', '',text, flags = re.IGNORECASE)
    text = re.sub(r'\btel\s*:*\s*\b', '',text, flags = re.IGNORECASE)
    text = re.sub(r'\bfax\s*:*\s*\b', '',text, flags = re.IGNORECASE)
    text = re.sub(r'\baddress\s*:*\s*\b', '',text, flags = re.IGNORECASE)
    text = re.sub(r'\bwebsite\s*:*\s*\b', '',text, flags = re.IGNORECASE)
    text = re.sub(r'\bimage\s*:*\s*\b', '',text, flags = re.IGNORECASE)
    #########
    # not necessary since all numbers are removed
    text = re.sub(r'((1-\d{3}-\d{3}-\d{4})|(\(\d{3}\) \d{3}-\d{4})|(\d{3}-\d{3}-\d{4}))$', '',text, flags = re.IGNORECASE)
    text = re.sub(r'\+(?=\d)', '',text, flags = re.IGNORECASE)
    return text

############################### TEXT PREPROCESSING ##################################
def count_word(text):
    '''
    Count number of words in an article
    
    Parameters
    -----------
    text: str
        text content of the article
        
    Returns
    -------
    word_count: int
        number of words in the article
    
    '''
    import re

    word_count = len(re.findall(r'\S+', text))
    return word_count


def get_category (url):
    '''
    Given the url of an article generate the first word in the path, 
    which is likely to be the categoryto which the article belongs to
    
    Parameters
    -----------
    url: str
        url of the article
        
    Returns
    -------
    category: str
        first word in the path component of the url
    
    '''
    import urllib.parse

    # parse the url to give the path part and access the first word in the path by splitting it by slash
    category = urllib.parse.urlparse(url)[2].split('/')[1]
    return category


def data_preprocessing(df):
    '''
    This function is used to preprocess and clean the input articles to remove irrelevant contents, thus ensure 
    the text data used to extract keyword are more business/finance/technology related
    
    Parameters
    -----------
    df: pandas dataframe
        article data, originally stored as csv file and input as pd dataframe
        
    Returns
    -------
    df_biz: pandas dataframe
        cleaned data that are considered to be more business/finance/technology related
    
    '''
    import pandas as pd
    import re
    
    # create a new column called text_length
    df['text_length'] = df['text'].apply(lambda row: count_word(row))
    # keep only articles that have more than 80 words
    df_clean = df[df.text_length > 80]
    
    # remove special characters apart from -%&/.,;:?! and whitespace
    df_clean['text'] = df_clean['text'].apply(lambda row: re.sub('[^a-zA-Z0-9-&/.,;:?!\s]', '', row))
    
    # remove articles that have access issues
    bad_text = ['Get Access', 'Get access', 'Buyouts Insider/Argosy Group LLC', 'AVCJ AVCJ', 'We use cookies on our website',
           'The Company is a publisher','free trail*', 'subscription option']
    df_clean = df_clean[~df_clean.text.str.contains('|'.join(bad_text))]
    
    #remove duplicate text
    df_clean2 = df_clean.drop_duplicates('text')
    
    # remove articles with title that contain trump, clinton, etc.
    df_notrump = df_clean2[~df_clean2.title.str.contains('Trump|Clinton|Obama|Bush|Syria|ISIS|Iran|Iraq|terrorist|terrorism|Terrorist|Terrorism')]
    
    # add category column given article's url
    df_notrump['category'] = df_notrump['url'].apply(lambda row: get_category(row))
    # remove non business content based on url parsing
    bad_cat = ['sport','sports','entertainment','politics','media','fashion','leisure','travel','environment',
          'health','media-news','fashion-news','style','life','Ingredients','weather','lifeandstyle',
          'menswear-news', 'leadership']
    df_biz = df_notrump[~df_notrump.category.isin(bad_cat)]
    
    return df_biz


############################### NOUN PHRASE EXTRACTION ##################################
def lambda_unpack(f):
    '''
    Needed by the get_nounphrase function
    
    '''
    return lambda args: f(*args)


def get_nounphrase(text, grammar = r'TT: {((<NN.*>|<JJ>) <NN.*>) | (<NNP>)}'):
    '''
    Extract noun phrases from the article using regular expression and nlp techniques
    
    Parameters
    -----------
    text: str
        text content of the article

    grammar: regex
        regular expression of noun phrases (Noun or Adjective, followed by Noun) and proper nouns
        
    Returns
    -------
    noun_phrases: list
        a list of noun phrases extracted from the article
    
    '''
    import re, itertools, nltk, string
    from nltk import word_tokenize, pos_tag, ne_chunk
    from nltk.tree import Tree

    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
 
    # Only convert candidates that are not Proper Noun Phrase to lowercase
    good_tags=set(['NNP','NNPS'])
    all_chunks_list = [list(elem) for elem in all_chunks]
    for chunks in all_chunks_list:
        if chunks[1] not in good_tags:
            chunks[0] = chunks[0].lower()
    all_chunks = [tuple(chunk) for chunk in all_chunks_list]
    
    # join constituent chunk words into a single chunked phrase and perform lemmatization
    lem = nltk.WordNetLemmatizer()
    candidates = [' '.join(lem.lemmatize(word) for word, pos, chunk in group) for key, group in itertools.groupby(all_chunks, lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]

    # exclude candidates that are entirely punctuation or stopwords
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    noun_phrases = [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]
    
    return noun_phrases


############################### KEYWORD EXTRACTION ##################################
def tf(word, blob):
    '''
    Frequency of the noun phrase, normalized by total number of phrases extracted
    
    Parameters
    -----------
    word: str
        the extracted noun phrase (minimum phrase length = 1)

    blob: list
        list of all the noun phrases extracted from the article
        
    Returns
    -------
    freq: float
        frequency of the noun phrase
    
    '''
    import collections

    d = collections.Counter(blob)
    freq = d[word] / len(blob)
    return freq

def n_containing(word, bloblist):
    '''
    Frequency of the noun phrase across the corpus (all articles in the given timeframe)
    
    Parameters
    -----------
    word: str
        the extracted noun phrase (minimum phrase length = 1)

    bloblist: list of list
        list of lists of noun phrases extracted from each article
        
    Returns
    -------
    overall_freq: float
        frequency of the noun phrase
    
    '''
    overall_freq = sum(1 for blob in bloblist if word in blob)
    return overall_freq

def idf(word, bloblist):
    '''
    Compute idf score (# of articles in the corpus divided by # of articles contain the noun phrase)
    
    Parameters
    -----------
    word: str
        the extracted noun phrase (minimum phrase length = 1)

    bloblist: list of list
        list of lists of noun phrases extracted from each article
        
    Returns
    -------
    idf_score: float
        idf score of the noun phrase
    
    '''
    idf_score = len(bloblist) / n_containing(word, bloblist)
    return idf_score

def tfidf(word, blob, bloblist):
    '''
    Compute tfidf score: tf * idf
    
    Parameters
    -----------
    word: str
        the extracted noun phrase (minimum phrase length = 1)

    blob: list
        list of all the noun phrases extracted from the article

    bloblist: list of list
        list of lists of noun phrases extracted from each article
        
    Returns
    -------
    tfidf_score: float
        tfidf score of the noun phrase
    
    '''
    tfidf_score = tf(word, blob) * idf(word, bloblist)
    return tfidf_score


def runmytfidf(df):
    '''
    Compute tfidf score: tf * idf
    
    Parameters
    -----------
    df: pandas dataframe
        processed article data
        
    Returns
    -------
    candidate_list: list of list
        a list of lists of keyword candidates from each article
        sorted in descending order of tfidf score
    
    '''
    from string import punctuation 
    
    NounPhrase = df['NounPhrase'].tolist()  
    
    candidate_list=[]
    for i, blob in enumerate(NounPhrase):
        # compute tfidf score for each noun phrase extracted from an articles
        scores = {word: tfidf(word, blob, NounPhrase) for word in blob}
        # sort them in descending order
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # remove single/double-letter word
        singlelist = [word for word,score in sorted_words if len(word.strip(punctuation)) > 2]
        candidate_list.append(singlelist)
    
    return candidate_list


def keyword_cutoff(full_list, word_count):
    '''
    Get different number of keywords based on the article length

    Parameters
    -----------
    full_list: list
        list of keyword candidates extracted from an article,
        sorted in descending tfidf score

    word_count: int
        number of words in the article
        
    Returns
    -------
    keyword_limit: list
        a list of keywords
    
    '''
    if word_count <= 250:
        keyword_limit = full_list[:10]

    elif 250 < word_count <= 1000:
        keyword_limit = full_list[:15]

    else:
        keyword_limit = full_list[:20]
    
    return keyword_limit


############################### KEYWORD POSTPROCESSING ##################################

def clean_keyword(keyword_list):
    '''
    Remove duplicate keywords that are in different format (in terms of capitalization, substring, order)
    but actually mean the same thing

    Parameters
    -----------
    keyword_list: list
        list of keywords
        
    Returns
    -------
    clean_keyword_ls: list
        clean list of keywords
    
    '''
    from string import punctuation
    from fuzzywuzzy import fuzz

    # keep only the titlecase, avoid having multiple words in different formats but actually means the same
    clean_ls = [item for item in keyword_list if item.istitle() or item.title() not in keyword_list]

    # remove leading and tailing punctuation and whitespace
    clean_ls2 = [item.strip(punctuation).strip() for item in clean_ls]

    # sort in ascending word length
    clean_ls3 = sorted(clean_ls2, key = len)

    # find substring and retain the longest term
    new = {}
    for i, word in enumerate(clean_ls3):
        new[word] = [word]
        for j in range(i + 1, len(clean_ls3)):
            if clean_ls3[i].lower() in clean_ls3[j].lower():
                new[word].append(clean_ls3[j])
        new[word] = sorted(new[word],key = len)[-1]

    final = list(set([value for key, value in new.items()]))
    
    # detect phrases with same set of words but arrange in different order
    for i, word in enumerate(sorted(final)):
        for j in range(i+1, len(final)):
            if fuzz.token_set_ratio(final[i], final[j]) == 100:
                final[i] = final[j]
    
    # keep unique items in the list
    clean_keyword_ls = list(set(final))
    
    return clean_keyword_ls


def one_row_format(df, lst_col):
    '''
    Convert dataframe into one-row-per-keyword format

    Parameters
    -----------
    df: pandas dataframe
        processed article data

    lst_col: str
        the name of the column that stored the list of keywords
        
    Returns
    -------
    df_inrow: pandas dataframe
        dataframe in a one-row-per-keyword format
    
    '''
    import numpy as np
    import pandas as pd

    df_inrow = pd.DataFrame({ 
    col:np.repeat(df[col].values, df[lst_col].str.len())
    for col in df.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(df[lst_col].values)})[df.columns.tolist()]

    return df_inrow


def get_titlecase(keyword):
    '''
    Turn keyword into title case (e.g. The Selected Keyword)

    Parameters
    -----------
    keyword: str
        keyword in its original form

    Returns
    -------
    tc_keyword: str
        keyword in title case form
    
    '''
    new_w = []
    for w in keyword.split():
        # avoid turning BMW to Bmw
        if w.isupper() is True:
            new_w.append(w)
        else:
            new_w.append(w.title())

    tc_keyword = (' '.join(new_w))

    return tc_keyword


def keyword_postprocessing(df_inrow):
    '''
    Remove keywords that are not relevant to the goal of the project(related to politics, sports, etc)
    and also those that are considered to be domain specific stop words

    Parameters
    -----------
    df_inrow: pandas dataframe
        dataframe in a one-row-per-keyword format

    Returns
    -------
    df_inrow_clean2: pandas dataframe
        dataframe in a one-row-per-keyword format without irrelevant keywords
    
    '''
    import pycountry
    import calendar
    # formal country names and their common alternative names/abbreviation
    ls = [country.name for country in pycountry.countries] # include 249 countries' official name
    alt_country = ['Syria','Iran','Korea','England','Britain','America','USA','U.S.A','U.S.','US','U.S','UK','U.K.','Isarel']
    ls.extend(alt_country)

    # time related words
    time_ls = [weekday for weekday in calendar.day_name] # weekday: Monday, Tuesday, etc.
    time_ls.extend([month for month in calendar.month_name[1:]]) # month: January, February, etc.
    time_ls.extend([m for m in calendar.month_abbr[1:]]) # month name abbreviation: Jan, Feb, Mar, etc.

    # Domain specific stop words
    others = ['London','New York','Los Angeles', 'San Francisco', 'Chicago', 'Tokyo','Hong Kong','Singapore',
              'Paris', 'Shanghai','Shenzhen','Sydney','Frankfurt', 'Europe','North America','Asia','Africa',
              'Management','Business','Market', 'Competition','Competitor','Acquisition','Partnership','Platform',
              'Organization', 'Corporation','Startup', 'Bank', 'Meeting','Conference', 'Industry', 'M&A', 'Merge & Acquisition',
              'Research','Development', 'Technology', 'Science','Operation', 'Promotion', 'Asset','Credit',
              'Interest', 'Capital','Capitalization','Investment','Deal', 'Agreement','Stakeholder',
              'IPO','Initial Public Offering','Series A','Series B','Series C', 'Deal','Investment', 'Transaction',
              'Series D', 'Seed','Seed Round', 'Angel Investor','Venture Capital','Private Equity',
              'Corporate Venture','Incubator','Accelerator','Investor','Angel','Valuation']

    ls.extend(time_ls)
    ls.extend(others)

    bad_kw = ['Year', 'Month','Quarter','Day','Morning','Evening','Afternoon', 'Date', 'Presentation', 'Loss', 'Issue','Plan',
    'Clinton','Trump','Obama','Election','Iran','Korea','Syria','Iraq','ISIS','U.S.','U.S','Terrorism','Terror','Terrorist',
    'Party', 'Government','Department','Ministry','Minister','President','Cabinet','Court', 'Bureau', 'Country','Society',
    'FBI','F.B.I','Office','Officer','Board', 'Headquarters','University','College', 'Instituition','Airport',
    'Story','Headline','World', 'Article', 'Statement','Report','Figure','Sheet','Period','Session','Terms', 'Strategy',
    'USD','GBP','JPY','CAD','EUR','CNY','AUD','NZD','CHF','XAU','XAG','Dollar','Crore','Pound','Euro', 'Money',
    'Percent','Cent', 'Profit','Income','Revenue','Click','Forex','Sale','Exchange','Stock','Share','Rate','Price',
    'Cost','Margin','Loan','Interest','Company','Companies','Executive','Olympic','Medal', 'Said','Assembly']

    bad_kw.extend(time_ls) # add in time related words

    # wrt keyword_clean_tc column
    # remove keywords that contain 'bad_kw'
    df_inrow_clean = df_inrow[~df_inrow.keyword_clean_tc.str.contains('|'.join(bad_kw))]
    # remove keywords if it is one of the words listed in the ls list
    df_inrow_clean2 = df_inrow_clean[~df_inrow_clean.keyword_clean_tc.isin(ls)]

    return df_inrow_clean2


############################### VERB PHRASE EXTRACTION ##################################
# This should be applied after trending keyword extraction to save computational cost
def get_verbphrase(text, keyword):

    from string import punctuation
    import nltk
    import re
    from nltk.corpus import stopwords
    
    stop_words = set(stopwords.words('english'))
    other_stopwords = ["according","including","also","too","though","however","nevertheless",
                       "could","would","yet","than"]
    stop_words.update(other_stopwords)
    sentences = [sent for sent in nltk.sent_tokenize(text)] # tokenize sentence
    pattern = re.compile('^\s+|\s*[,–;:?!:;()]\s*|\s+$') # separate by , : ; or () 
    r = re.compile('VB.*') # VB/VBD/VBG/VBN,etc. - verb in base form and other tenses
    
    # Extract parts of sentences in text that contain verb(s)
    phrase = []
    for sentence in sentences:
        part = pattern.split(sentence)
        for p in part:
            p = p.strip(punctuation) # remove both leading and tailing punctuation
            if len(p.split()) > 2: # exclude single word or NaN
                p_tag = nltk.pos_tag(nltk.word_tokenize(p)) # pos tagging
                tag_list = [tag for word, tag in p_tag]
                for t in tag_list:
                    if r.match(t): # see if there is a verb in the phrase
                        # add to list if there is verb in it, remove unicoding
                        phrase.append(p.replace('\r\n\r\n','')) 
                        break # stop once it encounters a verb
    
    # Extract verb phrases that contain the keyword
    verbphrase = []
    for vp in phrase:
        if keyword.isupper() is True:
            if keyword in vp:
                verbphrase.append(vp)
        else:
            if keyword.lower() in vp.lower():
                verbphrase.append(vp)
    
    # Clean the extracted verb phrase
    vp_clause = []
    if verbphrase:
        for vp in verbphrase:
            # break phrases
            vp_p = re.split(r'\bwhich\b|\bthat\b|\bwho\b|\bwhom\b|\bwhen\b|\bwhere\b|\bsaid\b|\bsays\b|\bsay\b', vp)
            if vp_p[-1]: 
                vp = vp_p[-1].strip() # keep the clause
            else:
                vp = vp_p[-2].strip()
            if vp:
                while vp.lower().split()[0] in stop_words: # remove leading stopwords if there are any
                    vp = ' '.join(vp.split()[1:])
                    if not vp or (vp.lower().split()[0] not in stop_words):
                        break
            if len(vp.split()) > 2: 
                vp_clause.append(vp)
    else:
        vp_clause = []
          
    clean_vp = []
    bad = ['say','says','said','report','reports','reported','claims','claim','claimed','add','adds','added',
          'comments','comment','commented','argue','argues','argued','suggest','suggests','suggested']
    if vp_clause:
        for vp_c in vp_clause:
            if vp_c.lower().split()[-1] not in bad: # remove verb phrases like ...XYZ says
                if vp_c.lower().split()[0] not in bad: # remove verb phrases like Says XYZ...
                    clean_vp.append(vp_c)
            
    else:
        clean_vp = []
    
    if len(clean_vp) == 0:
        clean_vp2 = None
    elif len(clean_vp) == 1:
        clean_vp2 = clean_vp
    elif len(clean_vp) > 1:
        clean_vp2 = [c_vp for c_vp in clean_vp if len(c_vp.split()) < 20]
        clean_vp2 = list(set(clean_vp2)) # keep only unique verb phrase

    
    # original_vp = len(clean_vp)
    
    return clean_vp2


############################### THE WHOLE PIPELINE ##################################
def overall (df):
    '''
    The whole pipeline in my part: from raw text to one-row-per-keyword format, filtering out irrelevant keywords
    The resulted dataframe will be used for trending keyword extraction.
    Trending keyword extraction should use the keywords in the 'keyword_clean_tc' column (in title case) 
    so all keywords are in consitent form

    Parameters
    -----------
    df: pandas dataframe
        dataframe that contains unprocessed article raw text
        column: text, url, title, timestamp

    Returns
    -------
    df_inrow_clean2: pandas dataframe
        dataframe with extracted keywords, ready to be used in trending keyword extraction
        column: text, url, title, timestamp, keyword_clean, keyword_clean_tc
    
    '''
    # text preprocessing, remove access request text and non-business articles in high level
    # create two columns: text length and category
    df_biz = data_preprocessing(df)

    # extract noun phrases from article, create a column called NounPhrase
    df_biz['NounPhrase'] = df_biz['text'].apply(lambda row: get_nounphrase(row))

    # sort noun phrases in order of descending tf-idf score, create a column named tfidf_full
    tfidfResult = pd.Series(runmytfidf(df_biz))
    df_biz['tfidf_full'] = tfidfResult.values

    # extract different number of keywords based on article length, create a column named keyword_limit
    df_biz['keyword_limit'] = df_biz.apply(lambda row: keyword_cutoff(row['tfidf_full'], row['text_length']), axis = 1)
    
    # remove duplicate keyword, store the rest of keywords in a new column keyword_clean
    df_biz['keyword_clean'] = df_biz['keyword_limit'].apply(lambda row: clean_keyword(row))

    # convert dataframe into one-row-per-keyword format
    df_inrow = one_row_format(df_biz, 'keyword_clean')

    # store the title case form of the keyword in a new column called keyword_clean_tc
    df_inrow['keyword_clean_tc'] = df_inrow['keyword_clean'].apply(lambda row: get_titlecase(row))

    # remove keywords that are not relevant to the goal of the project
    df_inrow_clean = keyword_postprocessing(df_inrow)

    # remove columns that are no longer needed
    col_not_needed = ['NounPhrase','tfidf_full','keyword_limit']
    df_inrow_clean2 = df_inrow_clean.drop(col_not_needed, axis=1)

    return df_inrow_clean2

















