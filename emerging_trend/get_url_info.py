def get_url_info(url):
    try:
        r = requests.head(url) 
        if r.status_code < 400: # if loads
            article = Article(url)
            article.download()
            article.parse()
            if detect(article.title) == 'en': #English only
                if len(article.text)>400: #filter out permission request
                    title = UnicodeDammit(article.title).unicode_markup
                    text = UnicodeDammit(article.text).unicode_markup
                    test_url= url
                    return title, text, test_url
    except Exception as e:
        print("fail ", datetime.datetime.now().time())
