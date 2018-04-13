def opt_title (candidates, doc_title):
    import inflect
    import collections
    from collections import OrderedDict
    candidate_scores = OrderedDict()
    
    for candidate in candidates:
        
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        engine = inflect.engine()
        plural = engine.plural(candidate)
        pattern2 = re.compile(r'\b'+re.escape(plural)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        
        # positional
        # found in title
        if pattern.search(doc_title):
            in_title = 1
        elif pattern2.search(doc_title):
            in_title = 1
        else:
            in_title = 0

        candidate_scores[candidate] = in_title
        
    in_title_list = []
    notin_title_list = []
    for term in candidate_scores.items():
        if term[1] == 1:
            in_title_list.append(term[0])
        else:
            notin_title_list.append(term[0])
    reranked = sorted(in_title_list) + notin_title_list
    
    return reranked

def opt_spread (candidates, doc_text, doc_title):
    import inflect
    from collections import OrderedDict
    candidate_scores = OrderedDict()
    
    for candidate in candidates:
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        
        # first/last position, difference between them (spread)
        doc_text_length = float(len(doc_text))
        first_match = pattern.search(doc_text)
        if not first_match:
             # convert back to plural form if the word is originally plural
            try:
                engine = inflect.engine()
                plural = engine.plural(candidate)
                pattern = re.compile(r'\b'+re.escape(plural)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
                first_match = pattern.search(doc_text)
            except IndexError:
                first_match = None
            if not first_match:
                spread = 0.0
            else:
                abs_first_occurrence = first_match.start() / doc_text_length
                if cand_doc_count == 1:
                    spread = 0.0
                    abs_last_occurrence = abs_first_occurrence
                else:
                    for last_match in pattern.finditer(doc_text):
                        pass
                    abs_last_occurrence = last_match.start() / doc_text_length
                    spread = abs_last_occurrence - abs_first_occurrence
        else:
            abs_first_occurrence = first_match.start() / doc_text_length
            if cand_doc_count == 1:
                spread = 0.0
                abs_last_occurrence = abs_first_occurrence
            else:
                for last_match in pattern.finditer(doc_text):
                    pass
                abs_last_occurrence = last_match.start() / doc_text_length
                spread = abs_last_occurrence - abs_first_occurrence

        candidate_scores[candidate] = spread
   
    #Rerank
    spread_list = []
    others = []
    for term in candidate_scores.items():
        if term[1] > 0:
            spread_list.append(term)
        else:
            others.append(term[0])
    spread_list_rank = sorted(spread_list,key=lambda x:(-x[1],x[0]))
    spread_list2 = []
    for t in spread_list_rank:
        spread_list2.append(t[0])
            
    final = spread_list2 + others

    return final

def opt_lexicalcohesion(candidates, doc_text):
    from collections import OrderedDict
    
    candidate_scores = OrderedDict()
    
    # get word counts for document
    doc_word_counts = collections.Counter(word.lower()
                                          for sent in nltk.sent_tokenize(doc_text)
                                          for word in nltk.word_tokenize(sent))
    
    for candidate in candidates:
        
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        
        # number of times candidate appears in document
        cand_doc_count = len(pattern.findall(doc_text))
        if not cand_doc_count:
            try:
                engine = inflect.engine()
                plural = engine.plural(candidate)
                pattern = re.compile(r'\b'+re.escape(plural)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
                cand_doc_count = len(pattern.findall(doc_text))
            except IndexError:
                cand_doc_count = 0
    
        # statistical
        candidate_words = candidate.split()
        max_word_length = max(len(w) for w in candidate_words)
        term_length = len(candidate_words)
        
        # get frequencies for term and constituent words
        sum_doc_word_counts = float(sum(doc_word_counts[w] for w in candidate_words))
        try:
            # lexical cohesion doesn't make sense for 1-word terms
            if term_length == 1:
                lexical_cohesion = 0.0
            else:
                lexical_cohesion = term_length * (1 + math.log(cand_doc_count, 10)) * cand_doc_count / sum_doc_word_counts
        except (ValueError, ZeroDivisionError) as e:
            lexical_cohesion = 0.0

        candidate_scores[candidate] = lexical_cohesion
    
    #Rerank
    lc_list = []
    others = []
    for term in candidate_scores.items():
        if term[1] > 0:
            lc_list.append(term)
        else:
            others.append(term[0])
    lc_list_rank = sorted(lc_list,key=lambda x:(-x[1],x[0]))
    lc_list2 = []
    for t in lc_list_rank:
        lc_list2.append(t[0])
            
    final = lc_list2 + others

    return final

def opt_title_lexical (candidates, doc_text, doc_title):
    from collections import OrderedDict
    
    candidate_scores = OrderedDict()
    
    # get word counts for document
    doc_word_counts = collections.Counter(word.lower()
                                          for sent in nltk.sent_tokenize(doc_text)
                                          for word in nltk.word_tokenize(sent))
    
    for candidate in candidates:
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        # positional
        # found in title
        in_title = 1 if pattern.search(doc_title) else 0
        
        # number of times candidate appears in document
        cand_doc_count = len(pattern.findall(doc_text))
        if not cand_doc_count:
            try:
                engine = inflect.engine()
                plural = engine.plural(candidate)
                pattern = re.compile(r'\b'+re.escape(plural)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
                cand_doc_count = len(pattern.findall(doc_text))
            except IndexError:
                cand_doc_count = 0
    
        # statistical
        candidate_words = candidate.split()
        max_word_length = max(len(w) for w in candidate_words)
        term_length = len(candidate_words)
        
        # get frequencies for term and constituent words
        sum_doc_word_counts = float(sum(doc_word_counts[w] for w in candidate_words))
        try:
            # lexical cohesion doesn't make sense for 1-word terms
            if term_length == 1:
                lexical_cohesion = 0.0
            else:
                lexical_cohesion = term_length * (1 + math.log(cand_doc_count, 10)) * cand_doc_count / sum_doc_word_counts
        except (ValueError, ZeroDivisionError) as e:
            lexical_cohesion = 0.0
        
        candidate_scores[candidate] = {'lexical_cohesion': lexical_cohesion,
                                       'in_title': in_title}
    # calculate score of each keyword
    rerank_dict = collections.OrderedDict()
    lc_score = []
    for term in candidate_scores.items():
        word = term[0]
        rerank_dict[word] = term[1]['in_title']
        lc_score.append(term[1]['lexical_cohesion'])

    # normalize lexical cohesion score to scale 0-1
    input_array=np.array(lc_score)
    result_array=(input_array-np.min(input_array))/np.ptp(input_array)
    # in case there is NaN in the array
    result_array[np.isnan(result_array)] = 0 
    i = 0
    for key, value in rerank_dict.items():
        rerank_dict[key] += result_array[i]
        i += 1
    
    # Rerank
    score_list = []
    others = []
    for term in rerank_dict.items():
        if term[1] > 0:
            score_list.append(term)
        else:
            others.append(term[0])
    score_list_rank = sorted(score_list,key=lambda x:(-x[1],x[0]))
    score_list2 = []
    for t in score_list_rank:
        score_list2.append(t[0])
            
    final = score_list2 + others

    return final

def opt_title_spread (candidates, doc_text, doc_title):
    from collections import OrderedDict
    
    candidate_scores = OrderedDict()
    
    for candidate in candidates:
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        
        # positional
        # found in title
        in_title = 1 if pattern.search(doc_title) else 0
        
        # first/last position, difference between them (spread)
        doc_text_length = float(len(doc_text))
        first_match = pattern.search(doc_text)
        if not first_match:
             # convert back to plural form if the word is originally plural
            try:
                engine = inflect.engine()
                plural = engine.plural(candidate)
                pattern = re.compile(r'\b'+re.escape(plural)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
                first_match = pattern.search(doc_text)
            except IndexError:
                first_match = None
            if not first_match:
                spread = 0.0
            else:
                abs_first_occurrence = first_match.start() / doc_text_length
                if cand_doc_count == 1:
                    spread = 0.0
                    abs_last_occurrence = abs_first_occurrence
                else:
                    for last_match in pattern.finditer(doc_text):
                        pass
                    abs_last_occurrence = last_match.start() / doc_text_length
                    spread = abs_last_occurrence - abs_first_occurrence
        else:
            abs_first_occurrence = first_match.start() / doc_text_length
            if cand_doc_count == 1:
                spread = 0.0
                abs_last_occurrence = abs_first_occurrence
            else:
                for last_match in pattern.finditer(doc_text):
                    pass
                abs_last_occurrence = last_match.start() / doc_text_length
                spread = abs_last_occurrence - abs_first_occurrence
        
        candidate_scores[candidate] = {'spread': spread,
                                       'in_title': in_title}
    # calculate score of each keyword
    rerank_dict = collections.OrderedDict()
    sp_score = []
    for term in candidate_scores.items():
        word = term[0]
        rerank_dict[word] = term[1]['in_title']
        sp_score.append(term[1]['spread'])

    # normalize lexical cohesion score to scale 0-1
    input_array=np.array(sp_score)
    result_array=(input_array-np.min(input_array))/np.ptp(input_array)
    # in case there is NaN in the array
    result_array[np.isnan(result_array)] = 0 
    i = 0
    for key, value in rerank_dict.items():
        rerank_dict[key] += result_array[i]
        i += 1

    # Rerank
    score_list = []
    others = []
    for term in rerank_dict.items():
        if term[1] > 0:
            score_list.append(term)
        else:
            others.append(term[0])
    score_list_rank = sorted(score_list,key=lambda x:(-x[1],x[0]))
    score_list2 = []
    for t in score_list_rank:
        score_list2.append(t[0])
            
    final = score_list2 + others

    return final

def opt_lexical_spread (candidates, doc_text, doc_title):
    from collections import OrderedDict
    
    candidate_scores = OrderedDict()
    
    # get word counts for document
    doc_word_counts = collections.Counter(word.lower()
                                          for sent in nltk.sent_tokenize(doc_text)
                                          for word in nltk.word_tokenize(sent))
    
    
    for candidate in candidates:
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        
        # Lexical Cohesion
        # number of times candidate appears in document
        cand_doc_count = len(pattern.findall(doc_text))
        if not cand_doc_count:
            try:
                engine = inflect.engine()
                plural = engine.plural(candidate)
                pattern = re.compile(r'\b'+re.escape(plural)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
                cand_doc_count = len(pattern.findall(doc_text))
            except IndexError:
                cand_doc_count = 0
    
        # statistical
        candidate_words = candidate.split()
        max_word_length = max(len(w) for w in candidate_words)
        term_length = len(candidate_words)
        
        # get frequencies for term and constituent words
        sum_doc_word_counts = float(sum(doc_word_counts[w] for w in candidate_words))
        try:
            # lexical cohesion doesn't make sense for 1-word terms
            if term_length == 1:
                lexical_cohesion = 0.0
            else:
                lexical_cohesion = term_length * (1 + math.log(cand_doc_count, 10)) * cand_doc_count / sum_doc_word_counts
        except (ValueError, ZeroDivisionError) as e:
            lexical_cohesion = 0.0
        
        # first/last position, difference between them (spread)
        doc_text_length = float(len(doc_text))
        first_match = pattern.search(doc_text)
        if not first_match:
             # convert back to plural form if the word is originally plural
            try:
                engine = inflect.engine()
                plural = engine.plural(candidate)
                pattern = re.compile(r'\b'+re.escape(plural)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
                first_match = pattern.search(doc_text)
            except IndexError:
                first_match = None
            if not first_match:
                spread = 0.0
            else:
                abs_first_occurrence = first_match.start() / doc_text_length
                if cand_doc_count == 1:
                    spread = 0.0
                    abs_last_occurrence = abs_first_occurrence
                else:
                    for last_match in pattern.finditer(doc_text):
                        pass
                    abs_last_occurrence = last_match.start() / doc_text_length
                    spread = abs_last_occurrence - abs_first_occurrence
        else:
            abs_first_occurrence = first_match.start() / doc_text_length
            if cand_doc_count == 1:
                spread = 0.0
                abs_last_occurrence = abs_first_occurrence
            else:
                for last_match in pattern.finditer(doc_text):
                    pass
                abs_last_occurrence = last_match.start() / doc_text_length
                spread = abs_last_occurrence - abs_first_occurrence
        
        candidate_scores[candidate] = {'spread': spread,
                                       'lexical cohesion': lexical_cohesion}

    # calculate score of each keyword
    rerank_dict = collections.OrderedDict()
    sp_score = []
    lc_score = []
    for term in candidate_scores.items():
        word = term[0]
        rerank_dict[word] = 0
        sp_score.append(term[1]['spread'])
        lc_score.append(term[1]['lexical cohesion'])

    # normalize lexical cohesion score to scale 0-1
    input_array1=np.array(lc_score)
    result_array1=(input_array1-np.min(input_array1))/np.ptp(input_array1)
    # in case there is NaN in the array
    result_array1[np.isnan(result_array1)] = 0 
    i = 0
    for key, value in rerank_dict.items():
        rerank_dict[key] += result_array1[i]
        i += 1

    # normalize spread score to scale 0-1
    input_array2=np.array(sp_score)
    result_array2=(input_array2-np.min(input_array2))/np.ptp(input_array2)
    # in case there is NaN in the array
    result_array2[np.isnan(result_array2)] = 0 
    i = 0
    for key, value in rerank_dict.items():
        rerank_dict[key] += result_array2[i]
        i += 1

    # Rerank
    score_list = []
    others = []
    for term in rerank_dict.items():
        if term[1] > 0:
            score_list.append(term)
        else:
            others.append(term[0])

    score_list_rank = sorted(score_list,key=lambda x:(-x[1],x[0]))
    score_list2 = []
    for t in score_list_rank:
        score_list2.append(t[0])
            
    final = score_list2 + others


    return final

def opt_title_lexical_spread (candidates, doc_text, doc_title):
    from collections import OrderedDict
    
    candidate_scores = OrderedDict()
    
    # get word counts for document
    doc_word_counts = collections.Counter(word.lower()
                                          for sent in nltk.sent_tokenize(doc_text)
                                          for word in nltk.word_tokenize(sent))
    
    
    for candidate in candidates:
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        
        # found in title
        in_title = 1 if pattern.search(doc_title) else 0
        
        # Lexical Cohesion
        # number of times candidate appears in document
        cand_doc_count = len(pattern.findall(doc_text))
        if not cand_doc_count:
            try:
                engine = inflect.engine()
                plural = engine.plural(candidate)
                pattern = re.compile(r'\b'+re.escape(plural)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
                cand_doc_count = len(pattern.findall(doc_text))
            except IndexError:
                cand_doc_count = 0
    
        # statistical
        candidate_words = candidate.split()
        max_word_length = max(len(w) for w in candidate_words)
        term_length = len(candidate_words)
        
        # get frequencies for term and constituent words
        sum_doc_word_counts = float(sum(doc_word_counts[w] for w in candidate_words))
        try:
            # lexical cohesion doesn't make sense for 1-word terms
            if term_length == 1:
                lexical_cohesion = 0.0
            else:
                lexical_cohesion = term_length * (1 + math.log(cand_doc_count, 10)) * cand_doc_count / sum_doc_word_counts
        except (ValueError, ZeroDivisionError) as e:
            lexical_cohesion = 0.0
        
        # first/last position, difference between them (spread)
        doc_text_length = float(len(doc_text))
        first_match = pattern.search(doc_text)
        if not first_match:
             # convert back to plural form if the word is originally plural
            try:
                engine = inflect.engine()
                plural = engine.plural(candidate)
                pattern = re.compile(r'\b'+re.escape(plural)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
                first_match = pattern.search(doc_text)
            except IndexError:
                first_match = None
            if not first_match:
                spread = 0.0
            else:
                abs_first_occurrence = first_match.start() / doc_text_length
                if cand_doc_count == 1:
                    spread = 0.0
                    abs_last_occurrence = abs_first_occurrence
                else:
                    for last_match in pattern.finditer(doc_text):
                        pass
                    abs_last_occurrence = last_match.start() / doc_text_length
                    spread = abs_last_occurrence - abs_first_occurrence
        else:
            abs_first_occurrence = first_match.start() / doc_text_length
            if cand_doc_count == 1:
                spread = 0.0
                abs_last_occurrence = abs_first_occurrence
            else:
                for last_match in pattern.finditer(doc_text):
                    pass
                abs_last_occurrence = last_match.start() / doc_text_length
                spread = abs_last_occurrence - abs_first_occurrence
        
        candidate_scores[candidate] = {'spread': spread,
                                       'lexical cohesion': lexical_cohesion,
                                      'in_title' : in_title}
    

    # calculate score of each keyword
    rerank_dict = collections.OrderedDict()
    sp_score = []
    lc_score = []
    for term in candidate_scores.items():
        word = term[0]
        rerank_dict[word] = term[1]['in_title']
        sp_score.append(term[1]['spread'])
        lc_score.append(term[1]['lexical cohesion'])

    # normalize lexical cohesion score to scale 0-1
    input_array1=np.array(lc_score)
    result_array1=(input_array1-np.min(input_array1))/np.ptp(input_array1)
    # in case there is NaN in the array
    result_array1[np.isnan(result_array1)] = 0 
    i = 0
    for key, value in rerank_dict.items():
        rerank_dict[key] += result_array1[i]
        i += 1

    # normalize spread score to scale 0-1
    input_array2=np.array(sp_score)
    result_array2=(input_array2-np.min(input_array2))/np.ptp(input_array2)
    # in case there is NaN in the array
    result_array2[np.isnan(result_array2)] = 0 
    i = 0
    for key, value in rerank_dict.items():
        rerank_dict[key] += result_array2[i]
        i += 1

    # Rerank
    score_list = []
    others = []
    for term in rerank_dict.items():
        if term[1] > 0:
            score_list.append(term)
        else:
            others.append(term[0])

    score_list_rank = sorted(score_list,key=lambda x:(-x[1],x[0]))
    score_list2 = []
    for t in score_list_rank:
        score_list2.append(t[0])
            
    final = score_list2 + others

    return final


abs_first_occurrence = 0.0
def opt_title_lexical_first (candidates, doc_text, doc_title):
    global abs_first_occurrence
    from collections import OrderedDict
    
    candidate_scores = OrderedDict()
    
    # get word counts for document
    doc_word_counts = collections.Counter(word.lower()
                                          for sent in nltk.sent_tokenize(doc_text)
                                          for word in nltk.word_tokenize(sent))
    
    
    for candidate in candidates:
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        
        # found in title
        in_title = 1 if pattern.search(doc_title) else 0
        
        # Lexical Cohesion
        # number of times candidate appears in document
        cand_doc_count = len(pattern.findall(doc_text))
        if not cand_doc_count:
            try:
                engine = inflect.engine()
                plural = engine.plural(candidate)
                pattern = re.compile(r'\b'+re.escape(plural)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
                cand_doc_count = len(pattern.findall(doc_text))
            except IndexError:
                cand_doc_count = 0
    
        # statistical
        candidate_words = candidate.split()
        max_word_length = max(len(w) for w in candidate_words)
        term_length = len(candidate_words)
        
        # get frequencies for term and constituent words
        sum_doc_word_counts = float(sum(doc_word_counts[w] for w in candidate_words))
        try:
            # lexical cohesion doesn't make sense for 1-word terms
            if term_length == 1:
                lexical_cohesion = 0.0
            else:
                lexical_cohesion = term_length * (1 + math.log(cand_doc_count, 10)) * cand_doc_count / sum_doc_word_counts
        except (ValueError, ZeroDivisionError) as e:
            lexical_cohesion = 0.0
        
        # first position
        doc_text_length = float(len(doc_text))
        first_match = pattern.search(doc_text)
        if not first_match:
             # convert back to plural form if the word is originally plural
            try:
                engine = inflect.engine()
                plural = engine.plural(candidate)
                pattern = re.compile(r'\b'+re.escape(plural)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
                first_match = pattern.search(doc_text)
            except IndexError:
                first_match = None
            if not first_match:
                abs_first_occurence = 0.0
            else:
                abs_first_occurrence = first_match.start() / doc_text_length
        else:
            abs_first_occurrence = first_match.start() / doc_text_length
        
        candidate_scores[candidate] = {'first': abs_first_occurrence,
                                       'lexical cohesion': lexical_cohesion,
                                      'in_title' : in_title}
    

    # calculate score of each keyword
    rerank_dict = collections.OrderedDict()
    sp_score = []
    lc_score = []
    for term in candidate_scores.items():
        word = term[0]
        rerank_dict[word] = term[1]['in_title']
        sp_score.append(term[1]['first'])
        lc_score.append(term[1]['lexical cohesion'])

    # normalize lexical cohesion score to scale 0-1
    input_array1=np.array(lc_score)
    result_array1=(input_array1-np.min(input_array1))/np.ptp(input_array1)
    # in case there is NaN in the array
    result_array1[np.isnan(result_array1)] = 0 
    i = 0
    for key, value in rerank_dict.items():
        rerank_dict[key] += result_array1[i]
        i += 1

    # normalize spread score to scale 0-1
    input_array2=np.array(sp_score)
    result_array2=(np.max(input_array2)-input_array2)/np.ptp(input_array2)
    # in case there is NaN in the array
    result_array2[np.isnan(result_array2)] = 0 
    i = 0
    for key, value in rerank_dict.items():
        rerank_dict[key] += result_array2[i]
        i += 1

    # Rerank
    score_list = []
    others = []
    for term in rerank_dict.items():
        if term[1] > 0:
            score_list.append(term)
        else:
            others.append(term[0])

    score_list_rank = sorted(score_list,key=lambda x:(-x[1],x[0]))
    score_list2 = []
    for t in score_list_rank:
        score_list2.append(t[0])
            
    final = score_list2 + others

    return final
