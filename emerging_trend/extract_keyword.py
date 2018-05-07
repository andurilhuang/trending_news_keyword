'''
- Laurie's part:
    -> Text Preprocessing
    -> Noun Phrase Extraction
    -> Keyword Extraction(tf, keyword processing)
    -> Data Structure Conversion
    ------------------------------
    -> Verb Phrase Extraction
    -> Keyword Clustering/Associated companies and investors

- The last function 'overall' goes through the whole pipeline in my part: from raw text to processed keyword 
  ready to be used in Su's trending keyword extraction


- Date: 05/06/2018

'''

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
    if len(urllib.parse.urlparse(url)[2].split('/')) > 1:
        category = urllib.parse.urlparse(url)[2].split('/')[1]
    else:
        category = None
    
    return category


def data_preprocessing(df, text):
    '''
    This function is used to preprocess and clean the input articles to remove irrelevant contents, thus ensure 
    the text data used to extract keyword are more business/finance/technology related
    
    Parameters
    -----------
    df: pandas dataframe
        article data, originally stored as csv file and input as pd dataframe

    text: str
        name of the column that stored the raw text
        
    Returns
    -------
    df_biz: pandas dataframe
        cleaned data that are considered to be more business/finance/technology related
    
    '''
    import pandas as pd
    import re
    
    # create a new column called text_length
    df['text_length'] = df[text].apply(lambda row: count_word(row))
    # keep only articles that have more than 80 words
    df_clean = df[df.text_length > 80]
    
    
    # remove articles that have access issues
    bad_text = ['Get Access', 'Get access', 'Buyouts Insider/Argosy Group LLC', 'AVCJ AVCJ', 'We use cookies on our website',
           'The Company is a publisher','free trail*', 'subscription option']
    df_clean = df_clean[~df_clean[text].str.contains('|'.join(bad_text))]
    
    #remove duplicate text
    df_clean2 = df_clean.drop_duplicates(text)
    
    # remove articles with title that contain trump, clinton, etc.
    df_notrump = df_clean2[~df_clean2.title.str.contains('Trump|Clinton|Obama|Bush|Syria|ISIS|Iran|Iraq|terrorist|terrorism|Terrorist|Terrorism|Terror')]
    
    # add category column given article's url
    df_notrump['category'] = df_notrump['url'].apply(lambda row: get_category(row))
    # remove non business content based on url parsing
    bad_cat = ['sport','sports','entertainment','politics','media','fashion','leisure','travel','environment',
          'health','media-news','fashion-news','style','life','Ingredients','weather','lifeandstyle',
          'menswear-news', 'leadership','Food-Safety','food']
    df_biz = df_notrump[~df_notrump.category.isin(bad_cat)]
    
    return df_biz

contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

def clean_text(text):
    '''
    This function is used to further clean the input articles to remove irrelevant words/suffixes
    
    Parameters
    -----------
    text: str
        raw text
        
    Returns
    -------
    text: str
        further cleaned text
    
    '''
    import re
    
    #remove CXO like words 
    for contraction in contractions.keys():
        text = text.replace(contraction,contractions[contraction]) #all contraction to formal forms
    cxo = ['CAO','CAIO','CIO','CFO','CTO','CEO','COO','CAE','CBO','CBDO','CCO','CDO','CEngO','CHRO','CMO','CNO','CPO','CRO','CVO']
    for term in cxo:
        text = text.replace(term, " ")   
    
    # remove various type of in-text urls/web link
    text = re.sub(r'http\S+', ' ',text, flags = re.IGNORECASE) # remove in text urls with http
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', text, flags=re.MULTILINE) # remove in text https
    text = re.sub(r'www.\S+', ' ',text, flags = re.IGNORECASE) #remove www urls
    text = re.sub(r'\S*@\S*\s?',' ',text, flags=re.MULTILINE) #remove in text email
    text = re.sub(r'\S*\.com\S*\s?',' ',text, flags=re.MULTILINE) #remove .com urls
    text = re.sub(r'\S*\.uk\S*\s?',' ',text, flags=re.MULTILINE) #remove .uk urls
    
    # date/time related
    text = re.sub(r'(\d)(st|nd|rd|th)',' ',text, flags=re.IGNORECASE)
    text = re.sub(r'\bPST\b|\bEST\b|\bGMT\b|\bCET\b|\ba\.?m\.?\b|\bp\.?m\.?\b', ' ',text, flags = re.IGNORECASE)
    text = re.sub(r'(\d)(am|pm|a.m.|p.m.|a.m.|p.m.)',' ',text, flags=re.IGNORECASE)
    
    # remove phone number
    text = re.sub(r'((1-\d{3}-\d{3}-\d{4})|(\(\d{3}\) \d{3}-\d{4})|(\d{3}-\d{3}-\d{4}))$', ' ',text, flags = re.IGNORECASE)
    # remove standealone number
    text = re.sub(r'\b\d+\b',' ',text, flags=re.MULTILINE)
    # remove special character
    text = re.sub("[^a-zA-Z0-9-&.–,’';:?!\s]", ' ', text)
    
    # remove additional, irrelevant information
    text = re.sub(r'mobile\s*:|phone\s*:|tel\s*:|fax\s*:|\btel\b|\bfax\b', ' ',text, flags = re.IGNORECASE)
    text = re.sub(r'address\s*:*|website\s*:|image\s*:|call\s*:', ' ',text, flags = re.IGNORECASE)
    text = re.sub(r'contact\s*:*|e*-*mail\s*:|source\s*:|ref\s?:|\bref\b|visit\s*:', ' ',text, flags = re.IGNORECASE)
    text = re.sub(r'place\s*:|figure\s*:|link\s*:|table\s*:', ' ',text, flags = re.IGNORECASE)
    

    # remove ',Inc.' type company suffix that the later cleaning step can't detect
    text = re.sub(r',\s*Inc\.', ' ',text)
    
    # unwanted word
    text = re.sub(r'\bplease\b|\babout\b|\bpercent\b|\bthanks\b|\bthank\b|\binclude\b|\bincluding\b|\bclick\b', ' ',text, flags = re.IGNORECASE)
    
    # remove extra whitespace
    text = re.sub( '\s+', ' ', text ).strip()
    
    return text



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

def get_keyword_candidates(NounPhrase, doc_title):
    '''
    Get keyword candidates from list of noun phrase by term frequency and if the phrase is in keyword
    - filter out city, US States and territories, country, region state_names
    - filter out domain specific stop words
    - filter out annual event (Super Bowl, Oscars, Grammys, etc.), politics word (Islam, terrorism, white house), natural disaster (flood, hurricane) 
    
    Parameters
    -----------
    NounPhrase: list
        a list of noun phrases extracted from the article

    title: str
        the title of the article where the noun phrases come from
        
    Returns
    -------
    reranked: list
        a list of keyword candidates
    '''
    
    import nltk
    import re
    import collections
    from fuzzywuzzy import fuzz
    from cleanco import cleanco
    
    # remove company suffix
    kw = []
    for np in NounPhrase:
        x = cleanco(np)
        kw.append(x.clean_name())
#     print(kw)
    kw0 = [re.sub('(Companys|Company|Companies|Firm|Organization|Corporation)','',k) for k in kw]
    kw0 = [k.strip() for k in kw0] # remove whitespace
    
    # remove leading, tailing, between-character punctuation
    punctuation ='’!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    kw1 =[]
    for k in kw0:
        k = ''.join(ch for ch in k if ch not in punctuation)
        if k and k != ' ': # remove empty string
            kw1.append(k)
#     print(kw1)
    
    # remove irrelevant keyword candidates
    ###### substring ######
    type1 = ['Percent','Major','Vote','Unit','Method','Option','Euro','Angel','Offer','Market','Review','Nation','Present','Direct','Terror','Islam','FY','Holder']
    
    kw2 = [word for word in kw1 if not any(w.lower() in word.lower() for w in type1)]
#     print(kw2)
    
    ###### subword ######
    import calendar
    import us
    from geotext import GeoText

    # time related words
    time_ls = [weekday for weekday in calendar.day_name]
    time_ls.extend([m for m in calendar.day_abbr])# weekday: Monday, Tuesday, etc.
    time_ls.extend([month for month in calendar.month_name[1:]]) # month: January, February, etc.
    time_ls.extend([m for m in calendar.month_abbr[1:]]) # month name abbreviation: Jan, Feb, Mar, etc.
    
    type2 = ['ISIS','Info','Cent','Part','RMB','EPS','SAR','IFRS','Plan','Deal','Time','Age','Rate','NYSE','GNP','REIT','Mr','Mrs','Ms','Co','Bn','Get','Bad','Dow','River','Lead','Employer','Difference',
             'ROI','AMEX','IRA','DJIA','NAV','PSP','FOREX','EFT','ETF','FDIC','FRB','LOI','NAV','SEC','YTM','NDA','SP','DC','etc','Zone','Such','SEK','Army','CFA','Net','Lake','Hotel','HKD','IST','Side',
             'EBITDA','FASB','FBMS','FDIC','GDP','BFY','OWCP','Gov','BLS','DOL','FDA','Site','EIS','Page','New','News','Old','Ltd','Corp','Task','Park','Esq','Tower','State','Return','War','Snow',
             'Sign','Step','Sale','NASDAQ','Job','No','CAGR','Discount','FBI','IRS','Cash','IRR','Tax','Taxation','Sir','Goal','Poor','Poors','ID','CPA','Hall','Stake','Association','Provision','Way',
             'Fact','Idea','Second','First','Half','Role','Big','Act','Share','DOJ','Sum','ASX','PhD','Line','Risk','Right','Rule','Read','See','TSX','Fed','IDF','NZX','Lot','Name','Soldier','Storm',
             'Loss','Gain','Person','Late','Team','Debt','Cost','Same','Last','Only','Area','Earnings','Earnings','Related','Performance','Palace','Temple','Stages','Inc','FTSE','Further','Rain','Investigation'
             'Year', 'Month','Quarter','Day','Morning','Evening','Afternoon', 'Date', 'Week','Hour','Minute','Period','Certain','Member','Republic','Prospect','Senate','Growth','Oscars','Source','Grammys',
             'Clinton','Trump','Obama','Election','Federal','Congress','Brand','Exchange','Authority','Requirement','Additional','Purchase','Esquire','Institute','Place','Crime','NBA','Available','EU',
             'Party', 'Government','Department','Ministry','Minister','President','Cabinet','Court','Bureau','Country','Society','Capitol','Assumption','Litte','Gross','Corporate','Said','Shares','UN',
             'Office','Officer','Board','Police','Law','Attorney','Analyst','Council','Street','Union','Branch','Request','Saving','Study','Expense','Strong','Per','Appendix','Billion','Competitive','Now',
             'Headquarters','University','College', 'Instituition','School','Academy','Airport','Station','Property','Avenue','Place','Quantity','Attachment','Next','Title','Yield','Ill','Sept','CAD','Top',
             'Statement','Statements','Report','Reports','Sheet','Sheets','Session','Term','Charter','Assessment','Application','Instruction','Publication','Period','Chapter','Weather','Times','EUR','Chair',
             'Document','Information','Transaction','Content','Press','Release','Journal','Form','Description','Section','Subsidiary','Attached','Editions','Relevant','Comment','Liquidity','Fortune','Free',
             'Agreement','Settlement','Filing','File','Award','Awards','Patent','Copyright','Strategy','Price','Asset','Factor','Documentation','Impact','Initiative','Several','Further','Choice','Sq','Ft',
             'Dividend','Profit','Income','Revenue','Margin','Interest','Influence','Problem','Securities','Currency','Great','Wrong','Claim','Proceeding','Strategic','Decision','Merge','Decline','Europe',
             'Security','Bond','Profile','Portfolio','Ratio','Rating','Value','Credit','Audit','Future','Instrument','Instruments','Policy','Other','Different','Expectation', 'Olympic','Decrease','Australia',
             'Finance','Financial','Financing','Component','Trade','Forecast','Prediction','Buy','Sell','Index','Staff','Concern','Expenditure','Justice','Edition','Inflation','Increase','Continue','Africa',
             'Business','Valuation','Series','Condition','Disclosure','Regulation','Committee','Rating','Stock','Excahnge','Quality','Spokesman','Competition','Serious','Average','Balance','Table','America',
             'Acquisition','Outlook','Prospectus','Stage','Executive','Budget','Investor','Owner','Leader','Acknowledgement','Overall','Competitor','Daily','Current','Medal','Buyouts','Allowance','Tsunami',
             'Announcement','Development','Account','Demand','Dollar','Crore','Pound','Number','Round','Many','Range','Relationship','Important','Chairman','Improvement','Allocation','Buyout','Flight','UK',
             'Assembly','Meeting','Conference','Access','Archive','Exhibit','Opportunity','Chance','Responsibility','Parameter','Later','Key','Hundred','Estimate','Phase','Judge','Governor','Asia','Drought',
             'Item','Product','Issue','Type','Class','Category','Amount','Result','Notes','Event','Order','Basis','Previous','Employee','Thousand','Summary','Chief','Position','Festival','Note','Earthquake',
             'Enquiry','Question','Answer','Reference','Action','Story','Headline','World','Article','Figure','Promotion','Certification','Level','Million','Notification','Principal','Did','Road','Flood','US',
             'High','Low','Total','Enough','Good','Recent','Annual','Above','Detail','Aggregate','Former','Manager','Effect','Thing','Standard','Deposit','Notice','Mortagage','Certificate','Agent','Hurricane']
    
    type2.extend(time_ls)
    
    kw3 = [word for word in kw2 if not any(w.lower() in word.lower().split() for w in type2)]
#     print(kw3)
    
    ###### the whole term ######
    type3 = ['Hongkong','Calif','Isarel','Korea','England','Britain','Agency','USA','U.S.A','U.S.','U.S','U.K.','UKs','UAE','Antarctica', 
    'Store','Silicon Valley','West','East','North','South','Northwest','Service Provider','Trust','Management','Partner','Program','Group',
    'Super Bowl','Limited Partner','General Partner','San','Southwest','Justice','Commerce','Head','Due Diligence','District','World',
    'Square Foot','PartnerSite','Parent Company','Don','Northeast','Partnership','Platform','Organization','Corporation','Startup', 'Bank', 
    'Industry', 'Sector','Segment','White House','Project','Research', 'Technology', 'Science','Operation', 'Capital','Capitalization','Investment',
    'IPO','Investment','Tech','Engineering','Fund','Seed','Venture Capital','Private Equity','Corporate Venture','Incubator','Accelerator','Customer',
    'Commission','Secretary','Client','Customer Service','Shop','Restaurant','City','Facility','Joint Venture','Website','Internet','Region','Function',
    'General','House','Appointment','Change','Founder','Author','Analysis','Full Text','Amendment','Venture','Free Online','Life','Treasury','Center']
    
    # any US state or territories name
    state_names = [state.name for state in us.states.STATES_AND_TERRITORIES]
    type3.extend(state_names)
    
    kw4 = [word for word in kw3 if word.lower() not in [w.lower() for w in type3]]
    # remove any country/city name
    kw4 = [word for word in kw4 if not GeoText(word.title()).countries if not GeoText(word.title()).cities]
#     print(kw4)    
    
    ###### Remove word length equals 1 and not noun/np term ######
    kw5 = []
    good_tag = ['NNP','NN']
    for word in kw4:
        if len(word.split()) == 1:
            for word, tag in nltk.pos_tag(word.split()):
                if tag in good_tag:
                    kw5.append(word)
        else:
            kw5.append(word)
#     print(kw5)
    
    # 'Global shares', 'global Shares','global shares','GLOBAL SHARES' -> 'Global Shares' if it exists in the list
    kw6 = [w.title() if w.title() in kw5 else w for w in kw5]
#     print(kw6)
    
    # count frequency
    c = collections.Counter(kw6)
    
    # 'Virtual Reality Technology','Virtual Reality','Technology' -> 'Virtual Reality Technology', 'Technology'
    kw6 = list(sorted(set(kw6), key = len, reverse = True)) # arrange in descending order of term length
    for i, word in enumerate(kw6):
        for j in range(i + 1, len(kw6)):
            if len(kw6[i].split()) == len(kw6[j].split()) and fuzz.token_set_ratio(kw6[i], kw6[j]) == 100:
                c[kw6[i]] += c[kw6[j]]
                del c[kw6[j]]
            elif len(kw6[i].split()) > len(kw6[j].split()) and kw6[i].split()[0].lower() == kw6[j].split()[0].lower():
                c[kw6[j]] += c[kw6[i]]
                del c[kw6[i]]

    # 'Technology','Virtual Reality Technology' -> 'Virtual Reality Technology'
    # 'Agency', 'Estate Agency','Real Estate Agency' -> 'Real Estate Agency'
    kw7 = sorted([key for key, value in c.items()], key = len) # arrange in ascending order of  term length
    for i, word in enumerate(kw7):
        for j in range(i + 1, len(kw7)):
            if len(kw7[i].split()) < len(kw7[j].split()) and kw7[i].split()[-1].lower() == kw7[j].split()[-1].lower():
                c[kw7[j]] += c[kw7[i]]
                del c[kw7[i]]
#     print(kw7)
    
    # Remove noun phrases with length greater than 3 words and shorter than 2 character
    # sort by occurring frequency
    sorted_kw = [key for key,value in sorted(c.items(), key=lambda x: x[1], reverse=True) if len(key)>2 and len(key.split()) < 4]
    
    # Rerank by giving more weight to terms occurred in the title
    kw_scores = collections.OrderedDict()
    
    for kw in sorted_kw:
        pattern = re.compile(r'\b'+re.escape(kw)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        if pattern.search(doc_title):
            in_title = 1
        else:
            in_title = 0

        kw_scores[kw] = in_title
  
    in_title_list = []
    notin_title_list = []
    for term in kw_scores.items():
        if term[1] == 1:
            in_title_list.append(term[0])
        else:
            notin_title_list.append(term[0])
    reranked = in_title_list + notin_title_list
    
    return reranked


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
        if len(full_list) < 5:
            keyword_limit = full_list
        else:
            keyword_limit = full_list[:5]

    elif 250 < word_count <= 1000:
        if len(full_list) < 10:
            keyword_limit = full_list
        else:
            keyword_limit = full_list[:10]

    else:
        if len(full_list) < 15:
            keyword_limit = full_list
        else:
            keyword_limit = full_list[:15]
    
    return keyword_limit


############################### KEYWORD POSTPROCESSING ##################################

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



############################### VERB PHRASE EXTRACTION ##################################
# This should be applied after trending keyword extraction to save computational cost
def get_verbphrase(text, keyword):
    '''
    Get verb phrases that contain the trending keyword from the raw text

    Parameters
    -----------
    text: list
        a list of raw text to which the trending keyword belongs

    keyword: str
        the trending keyword (in its original form)

    Returns
    -------
    vp_list: list
        list of lists of verb phrases that contain the trending keyword
    
    '''

    from string import punctuation
    import nltk
    import re
    from nltk.corpus import stopwords
    
    vp_list = []
    
    for t in text:
        stop_words = set(stopwords.words('english'))
        other_stopwords = ["according","including","also","too","though","however","nevertheless",
                           "could","would","yet","than"]
        stop_words.update(other_stopwords)
        sentences = [sent for sent in nltk.sent_tokenize(t)] # tokenize sentence
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
        
        vp_list.append(clean_vp2)

    
    return vp_list


############################### VERB PHRASE EXTRACTION ##################################
def get_industry_investor(pbid_list, lookup_tb):
    '''
    Get the industry and investors related to the trending keyword

    Parameters
    -----------
    pbid_list: list
        a list of pb entity id which the trending keyword associates with

    lookup_tb: dataframe
        the look up table provided by Pitchbook (named 'entity_industry_info.pkl')

    Returns
    -------
    final_series: series
        a series that contain investor, vertical and industry information
    
    '''
    import collections
    
    verticals = []
    investors = []
    industry = []
    
    for pb_id in pbid_list:
        # look up in the grouping table
        # find investors associated with each entity
        investors.append(list(lookup_tb.loc[lookup_tb['pbid'] == pb_id, "Active Investors"]))
        # identify the industry to which the entity belongs 
        industry.append(list(lookup_tb.loc[lookup_tb['pbid'] == pb_id, "industry"]))
        # identify the vertical to which the entity belongs 
        verticals.append(list(lookup_tb.loc[lookup_tb['pbid'] == pb_id, "Vertical"]))
    
    # sort the industry in descending order of associated entities 
    industry = [item for sublist in industry for item in sublist if item != None]
    counts = collections.Counter(industry)
    new_industry = list(set(sorted(industry, key=lambda x: -counts[x])))
    
    # get investors associated with all entities extracted
    investors = list(set([item for sublist in investors for item in sublist if str(item) != 'nan']))
    # get primary industry associated with all entities extracted
    industry = new_industry
    # get verticals associated with all entities extracted
    verticals = list(set([item for sublist in verticals for item in sublist if str(item) != 'nan']))
   
    if not investors:
        investors = None
    
    if not verticals:
        verticals = None
    
    if not industry:
        industry = None

    final_series = pd.Series((investors, verticals, industry))
    
    return final_series


############################### THE WHOLE PIPELINE ##################################
def keyword_extraction(df, raw_text,text):
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

    raw_text: str
        name of the column that stored the raw text
        
    text: str
        name of the column that stored the cleaned text

    Returns
    -------
    df_inrow_clean: pandas dataframe
        dataframe with extracted keywords, ready to be used in trending keyword extraction
        column: text, url, title, timestamp, keyword_clean, keyword_clean_tc
    
    '''
    import pandas as pd

    # text preprocessing, remove access request text and non-business articles in high level
    # create two columns: text length and category
    df_biz = data_preprocessing(df, raw_text)
    df_biz[text] = df_biz[raw_text].apply(lambda row: clean_text(row)) 
     

    print('Text preprocessing completed')

    # extract noun phrases from article, create a column called NounPhrase
    df_biz['NounPhrase'] = df_biz[text].apply(lambda row: get_nounphrase(row))

    print('Noun phrase extraction completed')

    # sort noun phrases in order of descending term frequency (reweighted by if the word is in title)
    # create a column called keyword_candidates
    df_biz['keyword_candidates'] = df_biz.apply(lambda row: get_keyword_candidates(row['NounPhrase'],row['title']), axis = 1)

    print('TF completed')

    # extract different number of keywords based on article length, create a column named keyword_clean
    df_biz['keyword_clean'] = df_biz.apply(lambda row: keyword_cutoff(row['keyword_candidates'], row['text_length']), axis = 1)

    print('Keyword extraction completed')

    # convert dataframe into one-row-per-keyword format
    df_inrow = one_row_format(df_biz, 'keyword_clean')

    # store the title case form of the keyword in a new column called keyword_clean_tc
    df_inrow['keyword_clean_tc'] = df_inrow['keyword_clean'].apply(lambda row: get_titlecase(row))

    print('Keyword titlecase completed')

    # remove columns that are no longer needed
    col_not_needed = ['NounPhrase','keyword_candidates']
    df_inrow_clean = df_inrow.drop(col_not_needed, axis=1)

    return df_inrow_clean


############################### SU TRENDING KEYWORD ##################################

def get_keyword_info(result_df, lookup_tb):
    '''
    Get the relevant verb phrase, investors, verticals and industry information related to each trending keyword
    Proceed after Su's part

    Parameters
    -----------
    result_df: pandas dataframe
        Trending keyword dataframe

    lookup_tb: str
        the look up table provided by Pitchbook (named 'entity_industry_info.pkl')


    Returns
    -------
    result_df: pandas dataframe
        dataframe with four extra columns: verb_phrase, investors, verticals and industry
    
    '''

    result_df['Verb_Phrase'] = result_df.apply(lambda row: get_verbphrase(row['text_raw'],row['Trending_keyword']), axis = 1)
    result_df[['Investors', 'Verticals','Industry']] = result_df.apply(lambda row: get_industry_investor(row['entityID'], lookup_table), axis=1)

    return result_df





