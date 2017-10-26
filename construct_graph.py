from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.lil import lil_matrix
from utils import *


def gen_cm(docs, preprocessor=None, tokenizer=word_tokenize, vocabulary=None, ngram_range=(1,1), min_df=10):
    count_model = CountVectorizer(preprocessor=preprocessor,
                                  tokenizer=tokenizer,
                                  vocabulary=vocabulary,
                                  ngram_range=ngram_range, lowercase=False, min_df=min_df)
    X = count_model.fit_transform(docs)
    Xc = (X.T * X)
    Xc.setdiag(0)
    return Xc, count_model.vocabulary_


def tokenize_docs():
    tokenized_docs = {}
    offline_data = pickle.load(open('data/offline_data.pkl', 'rb'))
    print('Data load complete, lemmatizing...')
    docs = [(str(doc['_id']), title_and_abstract(doc)) for doc in offline_data]
    voc = pickle.load(open('data/voc_30_index.pkl', 'rb'))
    print('Lemmatization complete, start tokenizing...')

    for _id, doc in docs:
        sentences = sent_tokenize(doc)
        tokenized_sentences = []
        for sent in sentences:
            tokenized_sentences.append(word_tokenize_dict(sent, voc))
        tokenized_docs[_id] = tokenized_sentences

    print('Complete, saving...')
    with open('data/tokenized_docs.pkl', 'wb') as f:
        pickle.dump(tokenized_docs, f)


def construct_graph():
    tokenized_docs = pickle.load(open('data/tokenized_docs.pkl', 'rb'))
    Xc, voc = gen_cm(flatten(tokenized_docs.values()), tokenizer=lambda x: x, preprocessor=lambda x: x)
    np.save('data/cm_graph', Xc)
    with open('data/generated_vocabulary.pkl', 'wb') as f:
        pickle.dump(voc, f)


def dump_graph():
    Xc = np.load('data/cm_graph.npy').item().todense()
    # voc = pickle.load(open('data/generated_vocabulary.pkl', 'rb'))
    print('Load complete')
    with open('data/cm_graph.txt', 'w') as f:
        for i in range(84918):
            for j in range(84918):
                c = Xc[i, j]
                if c > 0:
                    f.write(str(i) + '\t' + str(j) + '\t' + str(c) + '\n')


if __name__ == '__main__':
    # construct_graph()
    dump_graph()