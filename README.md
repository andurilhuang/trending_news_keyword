# trending_news_keyword

## Team Members
Students from University of Washington
Anna Huang, Ketong Lai, Qianru Wang , Su Wang

## Sponsor
Jenna Bono, Product Manager of PitchBook Data, Inc
  
## Organization Overview
PitchBook Data, Inc is a web-based financial data and software company that delivers data, research and technology covering the private capital markets, including venture capital, private equity and M&A transactions. The company’s core product is the PitchBook Platform, which is a subscription-only database that provides data and insights to help professionals in venture capital, private equity fund, investment banks and law firms understand the investment market and make decisions. Within the PitchBook Platform, users can also use a variety of software and analysis tools to run targeted searches, build financial models, create data visualizations, and build custom benchmarks. It also provides commentary and analysis of current events, trends and issues relevant to its field through PitchBook News and Analysis. 

## Project Overview
PitchBook has access to numerous sources of news articles and content. The sponsor associates these articles with firms and people using Machine Learning processes. The sponsor sees news as a potential indicator of change within a firm, sector, or market. 
Within a news article, keywords can help to identify the related entities and purpose of the article. These keywords would also be used to understand how topics or entities are “trending”. 

## Package Detials & Functions Highlight
*Crawling & DB Tool*

To crawl individual url info, see script "get_url" within package.

To crawl through and write into Dynamo large volumn of urls (>10k) see Jupyter Notebook "Get_Data_Write_Dynamo".

Refer to the notebook above for Dynamo writing procedures.




*Get Trending Keywords*

keyword_extraction:

The whole pipeline in my part: from raw text to one-row-per-keyword format, filtering out irrelevant keywords. The resulted dataframe will be used for trending keyword extraction.Trending keyword extraction should use the keywords in the 'keyword_clean_tc' column (in title case) so all keywords are in consitent form.

get_category: 

Given the url of an article generate the first word in the path, which is likely to be the categoryto which the article belongs to.

data_preprocessing: 

This function is used to preprocess and clean the input articles to remove irrelevant contents, thus ensure the text data used to extract keyword are more business/finance/technology related.

clean_text: 

This function is used to further clean the input articles to remove irrelevant words/suffixes.

get_nounphrase:

Extract noun phrases from the article using regular expression and nlp techniques.

get_keyword_candidates: 

Get keyword candidates from list of noun phrase by term frequency and if the phrase is in keyword.

keyword_cutoff: 

Get different number of keywords based on the article length.

get_verbphrase: 

Get verb phrases that contain the trending keyword from the raw text.

get_industry_investor: 

Get the industry and investors related to the trending keyword.

getEachKwHistoricalCountsDaily:

Calculate the number of articles that mention each keyword on each day within a given period.

getEachKwHistoricalCountsWeekly:

This function calculate the number of articles that mention each keyword in each week within a given period, if whole week can not be obtained within the duration given (i.e.22 days) the remaining days data will not be considered (i.e. only consider 21 days if duration = 22 ).

getArticleCoverageForKeyword:

Get each day's number of articles for a given keyword within a given period of time. For example an input of ('car',5, 20180423) will show the number of articles that mention keyword 'car'in the most recent 5 days for date 2018/04/23.

zscore:

This function calculates z score, which meansures the trend for a keyword, keywords with higher z score are more popular compared to others.
