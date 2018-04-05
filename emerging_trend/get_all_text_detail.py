#install all needed packages
import MySQLdb
from langdetect import detect
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from newspaper import Article
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from forex_python.converter import CurrencyCodes

def get_tier_3():
    """
    input: none
    output: import all urls into a list
    """
    import db_credit as dbc #info stored local with gitignore
    db = MySQLdb.connect(host= dbc.host,  
                         user= dbc.user,       
                         passwd= dbc.passwd,
                         port = dbc.port,
                         db= dbc.db)
     
    # Create a Cursor object to execute queries.
    cur = db.cursor()
     
    # Select data from table using SQL query.
    cur.execute(
                """
                select url from article
                """)
    
    url_list = []            
    urls = list(cur.fetchall())
    for url in urls:
        url_list.append(url[0])
    print ("url list gathering completed")
    return url_list
    
def get_continuous_chunks(text):
    """
    input: text string
    output: Entity (NER) recognized by NLTK
    type: str
    """
    c = CurrencyCodes() #currency transformation setup
    Clean_NER = []
    NER = []
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked: 
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            current_chunk = list(set(current_chunk)) #remove duplicates
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continue
    NER = list(set(continuous_chunk)) #remove duplicates 
    for i in NER:
        if c.get_symbol(i[:3]) is None:
            Clean_NER.append(i)
        
        NER = Clean_NER 

    return NER      
    
def get_url_info(url):
    """
    info: gets all detailed info per url
    input: url
    output: title, text, test_url, NER
    type: str
    """
    try:
        r = requests.head(url) 
        if r.status_code < 400: # if loads
            article = Article(url)
            article.download()
            article.parse()
            if detect(article.title) == 'en': #English only
                if len(article.text)>50: #filter out permission request
                    title = (article.title.encode('ascii', errors='ignore').decode("utf-8")) 
                    text = (article.text.encode('ascii', errors='ignore').decode("utf-8"))
                    test_url= url
                    NER = get_continuous_chunks(text)
                    #print ("success: ", url)
                    return title, text, test_url, NER
    except Exception as e:
        print(e, url) 

def append_url_info(url):
    """
    warning: this must be ran within the main combine function
    info: append individual url info into combined lists
    input: url
    output: title_list, text_list, test_urls, NER_list
    type: list
    """
    try:
        title, text, test_url, NER = get_url_info(url)
        title_list.append(title)
        text_list.append(text)
        test_urls.append(url)
        NER_list.append(NER)
        return title_list, text_list, test_urls, NER_list
    except (TypeError):
        pass        

def combine_text_info(url_num):
    """
    info: main call function to trigger all steps
    input: url_num (number of urls to crawl)
    output: pandas dataframe with all required info
    type: DataFrame
    """
    import time
    start = time.time()

    import threading
    #setting up buckets to catch tread call results
    title_list, text_list, test_urls, NER_list = [], [], [], []
    #get urls into list
    url_list = get_tier_3()[:url_num]
    #thread call
    threads = [threading.Thread(target=append_url_info, args=(url,)) for url in url_list]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    end = time.time()
    #timer ends
    duration = end - start
    #combine info into dictionary
    summary = {'url': test_urls, 'title': title_list, 'text': text_list, 'NER': NER_list}
    #transform dict into df
    summary_df = pd.DataFrame(summary)
    