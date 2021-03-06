from __future__ import print_function
import time
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
import pandas as pd
import datetime

import sys
sys.path.append('data_news/finBERT-master')

from finbert.finbert import predict
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

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
page_size = 1000
next_page = ''
company_ticker = []
name = []
article_id = []
date = []
time = []
title_summary = []
sentiment_score = []

collect_till_2019 = False

for ticker in tickers:
    while(True):
        response = intrinio.CompanyApi().get_company_news(ticker, page_size=page_size, next_page=next_page)
        for news_unit in response.news:
            print(news_unit.publication_date)
            if news_unit.publication_date.date().year == 2017:
                print('the year is 2019')
                collect_till_2019 = True
                break
            else:
                company_ticker.append(response.company.ticker)
                date.append(news_unit.publication_date.date())
                time.append(news_unit.publication_date.time())
                title_summary_item = news_unit.title + '. ' + news_unit.summary
                title_summary_item = title_summary_item.replace('\n', '').replace('\r', '')
                title_summary.append(title_summary_item)
                article_id.append(news_unit.id)
                name.append(response.company.name)
                ##### prediction lines ####
                with DisableLogger():
                    model_path = 'data_news/finBERT-master/models/sentiment/financial_phrasebank_pretrained'
                    model = BertForSequenceClassification.from_pretrained(model_path,num_labels=3,cache_dir=None)
                    output = predict(title_summary_item ,model)
                    print(output[['sentiment_score']])
                avg_sentiment_article = output[['sentiment_score']].mean(axis=0)
                sentiment_score.append(avg_sentiment_article)
                print('average')
                print(avg_sentiment_article)
        next_page = response.next_page
        if collect_till_2019 == True:
            break

dict = {'article_id': article_id, 'date': date, 'time': time,'ticker': company_ticker, 'name': name, 'title_summary': title_summary, 'sentiment_score': sentiment_score}
df = pd.DataFrame(dict)

df.to_csv('data_news/data_collection/aapl_news_data.csv')

