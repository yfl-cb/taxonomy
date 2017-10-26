import pickle
from collections import Counter
from pprint import pprint


tax = pickle.load(open('data/ACM_Taxonomy.pkl', 'rb'))
ctax = pickle.load(open('data/cleaned_acm_taxonomy.pkl', 'rb'))
rctax = {word: item for item, word in ctax.values()}
voc = pickle.load(open('data/generated_vocabulary.pkl', 'rb'))
rvoc = {v: k for k, v in voc.items()}
pairs = [line.strip().split('|') for line in open('data/cl2_pairs.txt')]
no_relations = [line.strip().split('|') for line in open('data/cl2_no_relation.txt')]


def root_of(item):
    while item['broader']:
        item = tax[item['broader'][0]]
    return item['prefLabel'].lower()


c = Counter()
cnt = 0
for k1, k2, _ in pairs:
    item1, item2 = rctax[rvoc[int(k1)]], rctax[rvoc[int(k2)]]
    root1, root2 = root_of(item1), root_of(item2)
    if root1 != root2:
        print(rvoc[int(k1)], '|', rvoc[int(k2)])
        print(root1, '|', root2)
        cnt += 1
    c[root1] += 1

pprint(c)

group1 = ['information systems', 'human-centered computing', 'security and privacy', 'mathematics of computing',
          'computer systems organization', 'theory of computation']

f1 = open('data/cl2_pairs_training.txt', 'w')
f2 = open('data/cl2_pairs_test.txt', 'w')
for k1, k2, w in pairs:
    item1, item2 = rctax[rvoc[int(k1)]], rctax[rvoc[int(k2)]]
    root1, root2 = root_of(item1), root_of(item2)
    if root1 in group1:
        f1.write('|'.join([k1, k2, str(w)]) + '\n')
    else:
        f2.write('|'.join([k1, k2, str(w)]) + '\n')


f1.close()
f2.close()

f1 = open('data/cl2_no_relation_training.txt', 'w')
f2 = open('data/cl2_no_relation_test.txt', 'w')
for k1, k2 in no_relations:
    item1, item2 = rctax[rvoc[int(k1)]], rctax[rvoc[int(k2)]]
    root1, root2 = root_of(item1), root_of(item2)
    if root1 in group1 and root2 in group1:
        f1.write('|'.join([k1, k2]) + '\n')
    elif root1 not in group1 and root2 not in group1:
        f2.write('|'.join([k1, k2]) + '\n')


training, testing = set(), set()
for line in [l for l in open('data/cl2_pairs_training.txt')] + [l for l in open('data/cl2_no_relation_training.txt')]:
    parts = line.strip().split('|')
    training.add(parts[0])
    training.add(parts[1])
for line in [l for l in open('data/cl2_pairs_test.txt')] + [l for l in open('data/cl2_no_relation_test.txt')]:
    parts = line.strip().split('|')
    testing.add(parts[0])
    testing.add(parts[1])
print(training)
print(testing)
print(len(training & testing))