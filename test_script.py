from sklearn.feature_extraction.text import CountVectorizer
from db_connection import *
from scipy.sparse.lil import lil_matrix
from utils import *
from collections import defaultdict as dd
from collections import Counter
import re, nltk


# ls = dd(set)
# c = Counter()
# offline_data = pickle.load(open('data/offline_data.pkl', 'rb'))
# for doc in offline_data:
#     for kw in doc['keywords']:
#         kw = lemmatize_long_word(kw)
#         # c[kw] += 1
#         ls[kw].add(str(doc['_id']))
#
# with open('data/voc_index.pkl', 'wb') as f:
#     pickle.dump(dict(ls), f)


# voc = pickle.load(open('data/voc.pkl', 'rb'))
# c = dd(set)
# for v in voc:
#     c[v] |= set([str(doc['_id']) for doc in pcol.find({'$text': {'$search': '" {} "'.format(v)}}, {'_id': 1})])
#
# with open('data/voc_abs.pkl', 'wb') as f:
#     pickle.dump(dict(c), f)

# pat = re.compile(r'\bincluding\b', re.I)
# cur = pcol.find({'abstract': pat})
# print(cur.count())
# such_as_texts = {}
# for doc in cur:
#     sentences = sent_tokenize(doc['abstract'])
#     targets = [s for s in sentences if re.search(pat, s)]
#     such_as_texts[str(doc['_id'])] = targets
#
# with open('data/including_pat_text.pkl', 'wb') as f:
#     pickle.dump(such_as_texts, f)

# 1. extract sentences
# 2. extract tokens that match the pattern
# 3. make it a partial taxonomy

PATTERN = 'such as'

voc_list = set(pickle.load(open('data/vocabulary.pkl', 'rb')))
voc = pickle.load(open('data/voc.pkl', 'rb'))

sentences = pickle.load(open('data/{}_pat_text.pkl'.format('_'.join(PATTERN.split())), 'rb'))
such_as_ids = set(sentences.keys())

# voc_with_pat = set(voc.keys()) | {'such as'}
voc_with_pat = (voc_list | {PATTERN}) - {'so'}
pat = re.compile(r'[,.-](.*?), {}(.+)[.]'.format(PATTERN))

excludes = {'application', 'operation', 'advantage', 'character'}

cnt, err = 0, 0
for _id, sentences in list(sentences.items()):
    for sent in sentences:
        # words = extract_words(sent, voc_with_pat)
        # print(words)
        # for former, latter in re.findall(pat, sent):
        #     former_w = extract_words(former, voc)
        #     latter_w = extract_words(latter, voc)
        #     if former_w and latter_w:
        #         print(former_w, latter_w)
        #         cnt += 1
        words = word_tokenize_dict(sent, voc_with_pat)
        try:
            pat_pos = words.index(PATTERN)
            broader = choose_broader(words, pat_pos)
            narrower = find_hyponyms(words, pat_pos)
            if broader in voc and in_vocab(narrower, voc) and len(broader.split()) > 0:
                narrower = [w for w in narrower if w in voc]
                print(sent)
                print(broader, '->', '; '.join(narrower))
                print('-------')
                cnt += 1
        except:
            err += 1
            continue

print(cnt, err)
