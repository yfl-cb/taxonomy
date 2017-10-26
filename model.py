from db_connection import *
from utils import *
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.lil import lil_matrix
from nltk import ngrams


def keywords_extraction(doc, using_original=True):
    if using_original:
        return [o.lower().strip() for o in doc['keywords']]

    return doc


def gen_cm(docs, preprocessor=None, tokenizer=word_tokenize, vocabulary=None, ngram_range=(1,1)):
    count_model = CountVectorizer(preprocessor=preprocessor,
                                  tokenizer=tokenizer,
                                  vocabulary=vocabulary,
                                  ngram_range=ngram_range, lowercase=True, min_df=2)
    X = count_model.fit_transform(docs)
    Xc = (X.T * X)
    Xc = lil_matrix(Xc)
    Xc.setdiag(0)
    return Xc, count_model.vocabulary_


def gen_cm_sentwise(docs, vocabulary=None):
    docs = flatten(map(lambda x: sent_tokenize(lemmatize_sentence(x['title']) + ' . ' + x['lem_abs']), docs))
    return gen_cm(docs, vocabulary=vocabulary, ngram_range=(1, 3))


def gen_cm_docwise(docs, vocabulary=None):
    return gen_cm([title_and_abstract(doc) for doc in docs], vocabulary=vocabulary, ngram_range=(1, 3))


def gen_cm_sentwind(docs, vocabulary=None, sent_ngram=None):
    def to_ngram_sentences(doc, n):
        sentences = sent_tokenize(lemmatize_sentence(' . '.join([doc['title'], doc['lem_abs']])))
        return [' '.join(tpl) for tpl in ngrams(sentences, n,
                                                pad_left=True, pad_right=True,
                                                left_pad_symbol='', right_pad_symbol='')]
    if sent_ngram is None:
        return gen_cm_docwise(docs, vocabulary=vocabulary)

    docs = flatten(map(lambda doc: to_ngram_sentences(doc, sent_ngram), docs))
    return gen_cm(docs, vocabulary=vocabulary, ngram_range=(1, 3))


if __name__ == '__main__':
    pcol = go_offline()
    TEST_LIMIT = 5000
    vocabulary = list(set(flatten([map(lambda x: lemmatize_long_word(x.lower()),
                                       o['keywords']) for o in pcol.find().limit(TEST_LIMIT)])))
    # with open('data/vocabulary.pkl', 'wb') as f:
    #     pickle.dump(vocabulary, f)
    vocabulary = pickle.load(open('data/voc.pkl', 'rb'))
    print(len(vocabulary))

    Xc, _ = gen_cm_sentwise(pcol.find().limit(TEST_LIMIT), vocabulary=vocabulary)
    np.save('data/cm_sentwise', Xc)
    # Xc, _ = gen_cm_docwise(pcol.find().limit(TEST_LIMIT), vocabulary=vocabulary)
    # np.save('data/cm_docwise', Xc)
    # Xc, voc = gen_cm_sentwind(pcol.find().limit(TEST_LIMIT), vocabulary=vocabulary, sent_ngram=3)
    # np.save('data/cm_sentwind_3', Xc)
    # Xc = np.load('data/cm_sentwind_3.npy')
    print(Xc)
