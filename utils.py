import nltk, pickle, itertools
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.data.path.append('/Users/yifan/Projects/taxonomy/nltk_data')


lemmatizer = WordNetLemmatizer()


def flatten(l):
    return [item for sublist in l for item in sublist]


def lemmatize_long_word(long_w):
    return ' '.join([lemmatizer.lemmatize(w) for w in long_w.lower().split()])


def title_and_abstract(doc):
    title_tokenized = ' '.join([lemmatizer.lemmatize(w.lower()) for w in word_tokenize(doc['title'])])
    return ' . '.join([title_tokenized, doc['lem_abs']])


def lemmatize_sentence(sent):
    return ' '.join([lemmatizer.lemmatize(w.lower()) for w in word_tokenize(sent)]).strip()


def extract_words(sentence, voc, ngram_range=(1, 4)):
    tokens = [lemmatizer.lemmatize(w.lower(), pos='n' if w != 'as' else 'v') for w in word_tokenize(sentence)]
    result, _s = [], set()
    for n in range(ngram_range[1], ngram_range[0] - 1, -1):
        for tok in nltk.ngrams(tokens, n, pad_right=True, pad_left=True, left_pad_symbol='', right_pad_symbol=''):
            tok = ' '.join(tok).strip()
            if tok in voc and tok not in _s:
                _s.add(tok)
                result.append(tok)
    return result


def word_tokenize_dict(sentence, voc, ngram_range=(1, 4)):
    tokens = [lemmatizer.lemmatize(w.lower(), pos='n' if w != 'as' else 'v') for w in word_tokenize(sentence)]
    result = []
    i = 0
    while i < len(tokens):
        flag = False
        for n in range(ngram_range[1], ngram_range[0], -1):
            tok = ' '.join(tokens[i: i + n])
            if tok in voc:
                result.append(tok)
                i += n
                flag = True
                break
        if not flag:
            result.append(tokens[i])
            i += 1
    return result


def find_hyponyms(seq, pat_pos):
    def isplit(iterable, splitters):
        return [list(g) for k, g in itertools.groupby(iterable, lambda x: x in splitters) if not k]

    result = []
    _seq = seq[pat_pos + 1:]
    stopsigns = {'so on', 'so', '.', 'which', 'etc', ',', 'are'}
    if 'and' in _seq:
        and_pos = _seq.index('and')
        # flag = False
        for i in range(and_pos + 1, len(_seq)):
            if _seq[i] in stopsigns:
                _seq = _seq[:i]
                # flag = True
                break

                # if not flag:
                #     _seq = _seq[:and_pos + 2]

    parts = isplit(_seq, (',', 'and'))
    for part in parts:
        result.append(' '.join(filter(lambda x: x not in stopsigns, part)))
    return result


def in_vocab(ls, voc):
    return any([w in voc for w in ls])


def choose_broader(seq, pat_pos):
    _seq = seq[:pat_pos - 1] if seq[pat_pos - 1] == ',' else seq[:pat_pos]
    for i in range(4):
        if _seq[-i] == 'of' and _seq[-i - 1] not in {'set', 'class'}:
            return _seq[-i - 1]
    return _seq


def title_abstract(item):
    return '. '.join([item.get('title', ''), item.get('abstract', '')])
