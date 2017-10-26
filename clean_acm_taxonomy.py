from utils import *
import codecs


tax = pickle.load(open('data/ACM_Taxonomy.pkl', 'rb'))
voc = pickle.load(open('data/generated_vocabulary.pkl', 'rb'))

exclude_meta_categories = ['general and reference', 'social and professional topics',
                           'proper nouns: people, technologies and companies']


def root_of(item):
    while item['broader']:
        item = tax[item['broader'][0]]
    return item['prefLabel'].lower()


def in_voc(item):
    w = lemmatize_long_word(item['prefLabel'].lower())
    if w in voc:
        return w
    for alt in item['altLabel']:
        w = lemmatize_long_word(alt.lower())
        if w in voc:
            return w
    return None


# cleaned = {}
# for kid, item in tax.items():
#     if root_of(item) in exclude_meta_categories:
#         continue
#
#     w = in_voc(item)
#     if not w:
#         print(kid, item['prefLabel'].replace('\xf6', ''))
#         continue
#
#     cleaned[kid] = item, w
#
# print(len(cleaned))

# with open('data/cleaned_acm_taxonomy.pkl', 'wb') as f:
#     pickle.dump(cleaned, f)

rtax = {item['prefLabel'] for item in tax.values()}
ctax = pickle.load(open('data/cleaned_acm_taxonomy.pkl', 'rb'))
rctax = {item['prefLabel']: (item, word) for item, word in ctax.values()}


with codecs.open('data/cl_pairs_text.txt', 'w', 'utf8') as f:
    for line in codecs.open('data/pairs.txt', 'r', 'utf8'):
        tpl = line.split('|')
        k1, k2, w = tpl[0], tpl[1], int(tpl[2])
        if k1 in rctax and k2 in rctax:
            f.write('|'.join([str(rctax[k1][1]), str(rctax[k2][1]), str(w)]) + '\n')

with codecs.open('data/cl_no_relation_text.txt', 'w', 'utf8') as f:
    for line in codecs.open('data/no_relation.txt', 'r', 'utf8'):
        tpl = line.split('|')
        k1, k2 = tpl[0].strip(), tpl[1].strip()
        if k1 in rctax and k2 in rctax:
            f.write('|'.join([str(rctax[k1][1]), str(rctax[k2][1])]) + '\n')
