from pymongo import MongoClient
from bson import ObjectId
import pickle


client = MongoClient('166.111.7.173:11060')
db = client['sttc']

pcol = db['comp_pubs_extracted']
icol = db['acm_keywords']

text_cache = {line.split('|')[0]: line.split('|')[1].split(',') for line in open('data/text_cache.txt')}
offline_data = None


class OfflineCursor:
    def __init__(self, ls):
        self.ls = ls

    def find(self):
        return self

    def limit(self, n):
        return self

    def __iter__(self):
        for item in self.ls:
            yield item


def pcol_text_search(_s):
    if _s in text_cache:
        return pcol.find({'_id': {'$in': [ObjectId(_id) for _id in text_cache[_s]]}})
    res = list(pcol.find({'$text': {'$search': _s}}, {'_id': 1}))
    text_cache[_s] = [str(doc['_id']) for doc in res]
    with open('data/text_cache.txt', 'a') as f:
        f.write('{}|{}\n'.format(_s, ','.join(text_cache[_s])))
    return res


def pcol_text_search_count(_s):
    if _s in text_cache:
        return len(text_cache[_s])
    return len(pcol_text_search(_s))


def pcol_find(**kwargs):
    return pcol.find(**kwargs)


def go_offline():
    return OfflineCursor(pickle.load(open('data/offline_data.pkl', 'rb')))

