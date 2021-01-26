#%%
from __future__ import print_function
import time
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
import pandas as pd
import datetime

import sys
sys.path.append('data_news/finBERT-master')

print(sys.path)

#%%
from finbert.finbert import predict
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
#%%
import logging
class DisableLogger():
                def __enter__(self):
                    logging.disable(logging.WARNING)
                def __exit__(self, exit_type, exit_value, exit_traceback):
                    logging.disable(logging.NOTSET)

#sandbox key    
intrinio.ApiClient().set_api_key('OjI0ZmZjNDNmOTlmZDhmZDgzM2VlMzdlYjFiZDAzZjIx')
#production key
# intrinio.ApiClient().set_api_key('OmViZmZjMDA4Njg2YmM1ZTAzYWI0ZDEyZjk0Mjk3NTE1')
intrinio.ApiClient().allow_retries(True)

tickers = ['AAPL']
page_size = 2
next_page = ''
company_ticker = []
name = []
article_id = []
date = []
time = []
title_summary = []
sentiment_score = []

x = 5
collect_till_2017 = False


for ticker in tickers:
    while(True):
        response = intrinio.CompanyApi().get_company_news(ticker, page_size=page_size, next_page=next_page)
        for news_unit in response.news:
            if x == 6:
                collect_till_2017 = True
                break
            else:
                print('unit')
                company_ticker.append(response.company.ticker)
                date.append(news_unit.publication_date.date())
                time.append(news_unit.publication_date.time())
                title_summary.append(news_unit.title + '. ' + news_unit.summary)
                article_id.append(news_unit.id)
                name.append(response.company.name)
##### prediction lines ####
                with DisableLogger():
                    model_path = 'data_news/finBERT-master/models/sentiment/financial_phrasebank_pretrained'
                    model = BertForSequenceClassification.from_pretrained(model_path,num_labels=3,cache_dir=None)
                    text = news_unit.title + '. ' + news_unit.summary
                    text = text.replace('\n', ' ').replace('\r', '')
                    output = predict(text,model)
                    print(output[['sentiment_score']])
                avg_sentiment_article = output[['sentiment_score']].mean(axis=0)
                sentiment_score.append(avg_sentiment_article)
                print('average')
                print(avg_sentiment_article)
                x = x + 1
        if collect_till_2017 == True:
            break

dict = {'article_id': article_id, 'date': date, 'time': time,'ticker': company_ticker, 'name': name, 'title_summary': title_summary, 'sentiment_score': sentiment_score}
df = pd.DataFrame(dict)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns',20)
# print(df) 
# print(df.loc[0, ['title_summary']])
######################
