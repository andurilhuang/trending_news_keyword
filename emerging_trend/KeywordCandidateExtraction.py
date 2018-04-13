def lambda_unpack(f):
    return lambda args: f(*args)


def keyword_candidate_technicalterm(text, grammar = r'TT: {(<JJ.*>|<NN.*>)+ (<NN.*>|<CD>)|<NN.*>}'):
    import itertools, nltk, string
    # exclude candidates that are entirely punctuation or stopwords
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
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
    
    # join constituent chunk words into a single chunked phrase, change them into lowercase, lemmatization
    lem = nltk.WordNetLemmatizer()
    candidates = [' '.join(word for word, pos, chunk in group) for key, group in itertools.groupby(all_chunks, lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]
    candidates_new = []
    for cand in candidates:
        constituent = []
        for c in cand.split():
            if c[0].isupper() is False:
                constituent.append(lem.lemmatize(c))
            else:
                constituent.append(c)
        candidates_new.append(' '.join(con for con in constituent if con not in punct))

    return [cand for cand in candidates_new
            if cand not in stop_words and not all(char in punct for char in cand)]

def keyword_candidate_technicalterm_combined(text, grammar = r'TT: {(<JJ.*>|<NN.*>)+ (<NN.*>|<CD>)|<NN.*>}'):
    
    # exclude candidates that are entirely punctuation or stopwords
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    
    candidates = [' '.join(word for word, pos, chunk in group) for key, group in itertools.groupby(all_chunks, lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]
    
    original = [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]
    
    # Only convert candidates that are not Proper Noun Phrase to lowercase
    good_tags=set(['NNP','NNPS'])
    all_chunks_list = [list(elem) for elem in all_chunks]
    for chunks in all_chunks_list:
        if chunks[1] not in good_tags:
            chunks[0] = chunks[0].lower()
    all_chunks = [tuple(chunk) for chunk in all_chunks_list]
    
    # lemmatization
    lem = nltk.WordNetLemmatizer()
    candidates_new = []
    for cand in candidates:
        constituent = []
        for c in cand.split():
            if c[0].isupper() is False:
                constituent.append(lem.lemmatize(c))
            else:
                constituent.append(c)
        candidates_new.append(' '.join(con for con in constituent if con not in punct))
    
    processed = [cand for cand in candidates_new if cand not in stop_words and not all(char in punct for char in cand)]
    
    original_processed = dict(zip(original, processed))
    
    return original, processed, original_processed

def keyword_candidate_technicalterm2(text, grammar = r'TT: {(<JJ.*>|<NN.*>)+ (<NN.*>|<CD>)|<NN.*>}'):
    
    # exclude candidates that are entirely punctuation or stopwords
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    
    
    # join constituent chunk words into a single chunked phrase, change them into lowercase, lemmatization
    candidates = [' '.join(word for word, pos, chunk in group) for key, group in itertools.groupby(all_chunks, lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]


def keyword_candidate_nounphrase(text, grammar = r'NP: {(<JJ.*>* <NN.*>+)? <JJ.*>* <NN.*>+}'):
    import itertools, nltk, string
    # exclude candidates that are entirely punctuation or stopwords
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
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
    
    # join constituent chunk words into a single chunked phrase, change them into lowercase, lemmatization
    lem = nltk.WordNetLemmatizer()
    candidates = [' '.join(word for word, pos, chunk in group) for key, group in itertools.groupby(all_chunks, lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]
    candidates_new = []
    for cand in candidates:
        constituent = []
        for c in cand.split():
            if c[0].isupper() is False:
                constituent.append(lem.lemmatize(c))
            else:
                constituent.append(c)
        candidates_new.append(' '.join(con for con in constituent if con not in punct))
        
    return [cand for cand in candidates_new
            if cand not in stop_words and not all(char in punct for char in cand)]


def keyword_candidate_verbphrase(text, grammar = r'VP: {(<JJ.*>|<NN.*>)+ (<NN.*>|<CD>)|<NN.*>}'):
    
    # exclude candidates that are entirely punctuation or stopwords
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group) for key, group in itertools.groupby(all_chunks, lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]
    
    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]