def table_to_df():
    """
    Function gets the whole Dynabo table into a dataframe
    Input: none
    Output: dataframe of raw data from Dynamo
    Output type: 
    """
    import boto3
    import pandas as pd

    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

    table = dynamodb.Table('url_text')
    response = table.scan()
    data = response['Items']

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    
    title_list, text_raw_list, text_clean_list, test_urls, NER_list, time_list = [], [], [], [], [], []
    
    for item in data:
        NER_list.append(item['NER'])
        title_list.append(item['title'])
        text_raw_list.append(item['text_raw'])
        text_clean_list.append(item['text_clean'])
        time_list.append(item['timestamp'])
        test_urls.append(item['url'])

    summary = {'url': test_urls, 'title': title_list, 'text_raw': text_raw_list, 'text_clean':text_clean_list, 'NER': NER_list, 'timestamp': time_list}
    summary_df = pd.DataFrame(summary)    
    
    return summary_df   
    
def get_dynamo():
    """
    Function match dynamo dataframe with entity name an id, return a roughly cleaned and updated dataframe
    Input: None
    Output: cleaned and updated dataframe
    Output type: dataframe
    """
    import MySQLdb
    import db_credit as dbc
    import pandas as pd
    import re

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
                select pb.pbid
                from tier_3_output.article a 
                join tier_3_output.article_to_rts_entities  atr
                on  a.article_id = atr.article_id
                join tier_3_output.pbid_businessentity pb
                on atr.rts_entity_id = pb.entity_id
                where a.timestamp between '2016-12-01 00:00:00' and '2017-02-15 23:00:00'
                """)    
    id_list = []            
    ids = list(cur.fetchall())

    for idi in ids:
        id_list.append(idi[0])

    cur.execute(
                """
                select pb.entity_name
                from tier_3_output.article a 
                join tier_3_output.article_to_rts_entities  atr
                on  a.article_id = atr.article_id
                join tier_3_output.pbid_businessentity pb
                on atr.rts_entity_id = pb.entity_id
                where a.timestamp between '2016-12-01 00:00:00' and '2017-02-15 23:00:00'
                """)    
    entity_list = []            
    entities = list(cur.fetchall())

    for entity in entities:
        entity_list.append(entity[0])

    cur.execute(
                """
                select a.url
                from tier_3_output.article a 
                join tier_3_output.article_to_rts_entities  atr
                on  a.article_id = atr.article_id
                join tier_3_output.pbid_businessentity pb
                on atr.rts_entity_id = pb.entity_id
                where a.timestamp between '2016-12-01 00:00:00' and '2017-02-15 23:00:00'
                """)    
    url_list = []            
    urls = list(cur.fetchall())

    for url in urls:
        url_list.append(url[0])    

    root_dict = {'url_list': url_list, 'id_list':id_list, 'entity_list':entity_list}

    root_table = pd.DataFrame(root_dict)
    root_table.head()

    unique_urls = set(url_list)

    root_entity = root_table.groupby('url_list')['entity_list'].apply(list)
    entity_root = root_entity.reset_index()
    entity_root = entity_root.set_index('url_list')

    root_id = root_table.groupby('url_list')['id_list'].apply(list)
    id_root = root_id.reset_index()
    id_root = id_root.set_index('url_list')

    entity_list_matched = []
    pbid_list_matched = []

    summary_df = table_to_df()
    summary_df['text_clean'] = [re.sub( '\s+', ' ', text ).strip() for text in summary_df['text_clean']]
    
    for url in summary_df['url']:
        if url in unique_urls:
            entity_list_matched.append(list(entity_root.loc[url])[0])
            pbid_list_matched.append(list(id_root.loc[url])[0])
        else:
            entity_list_matched.append('NA')
            pbid_list_matched.append('NA')

    summary_df['entity'] = entity_list_matched
    summary_df['pbid'] = pbid_list_matched 
    
    return summary_df    
