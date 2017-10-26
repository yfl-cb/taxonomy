from db_connection import *
from urllib.request import urlopen, Request
import json, sys


def generate_request(url, data=None):
    return Request(
        url=url,
        data=data.encode('utf-8'),
    )


def get_raw_content_of_url(url, data=None):
    with urlopen(generate_request(url, data)) as response:
        html = response.read().decode('utf-8')
    return html


url = 'http://yutao.yt:9000/?properties=%7B%22annotators%22%3A%20%22tokenize%2Cssplit%2Cpos%2Cner%2Cdepparse%2Copenie%22%2C%20%22date%22%3A%20%222017-09-20T10%3A49%3A33%22%7D&pipelineLanguage=en'

for doc in pcol.find({}, {'abstract': 1}).limit(2):
    try:
        content = json.loads(get_raw_content_of_url(url, doc['abstract']))
        pcol.update_one({'_id': doc['_id']}, {'$set': {'corenlp': content}})
    except Exception as e:
        print(str(doc['_id']), e, file=sys.stderr)
