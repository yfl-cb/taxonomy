from utils import *


voc = pickle.load(open('data/voc_abs.pkl', 'rb'))
offline_data = {str(doc['_id']): doc for doc in pickle.load(open('data/offline_data.pkl', 'rb'))}

wordpair_dict = {}
cnt = 0
for w1, w2 in itertools.combinations(voc.keys(), 2):
    intersect = voc[w1] & voc[w2]
    cnt += 1
    if cnt % 1000 == 0:
        print(cnt)
    for _id in intersect:
        lem_abs = offline_data[_id]['lem_abs']
        sentences = list(filter(lambda x: w1 in x and w2 in x, sent_tokenize(lem_abs)))
        if sentences:
            wordpair_dict[_id] = sentences

with open('data/wordpair_dict.pkl', 'wb') as f:
    pickle.dump(wordpair_dict, f)
