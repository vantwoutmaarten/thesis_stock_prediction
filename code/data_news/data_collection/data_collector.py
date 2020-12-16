from __future__ import print_function
import time
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
import pandas as pd

intrinio.ApiClient().set_api_key('OjI0ZmZjNDNmOTlmZDhmZDgzM2VlMzdlYjFiZDAzZjIx')
intrinio.ApiClient().allow_retries(True)

identifier = 'AAPL'
page_size = 10
next_page = ''

response = intrinio.CompanyApi().get_company_news(identifier, page_size=page_size, next_page=next_page)
print(response)


