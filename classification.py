from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from random import shuffle
import keras.backend as K


embedding = {}


def read_embedding():
    for line in open('data/vec_all.txt'):
        parts = line.split()
        embedding[int(parts[0])] = [float(i) for i in parts[1:]]


def split(arr):
    return [i[0] for i in arr], [i[1] for i in arr]


def read_data(test_percent=0.5):
    ps_tr, ps_ts, ns_tr, ns_ts = [], [], [], []
    # for line in open('data/cl_combine.txt'):
    #     parts = line.split('|')
    #     k1, k2 = int(parts[0]), int(parts[1])
    #     if len(parts) < 3 or int(parts[2]) == 0:
    #         ns.append((embedding[k1] + embedding[k2], 0))
    #     else:
    #         if int(parts[2]) < 0:
    #             k1, k2 = k2, k1
    #         ps.append((embedding[k1] + embedding[k2], 1))


    training_ws, testing_ws = set(), set()
    for line in open('data/cl2_pairs_training.txt'):
        parts = line.split('|')
        k1, k2, w = int(parts[0]), int(parts[1]), float(parts[2])
        training_ws.add(k1)
        training_ws.add(k2)
        if w < 0:
            k1, k2 = k2, k1
        ps_tr.append((embedding[k1] + embedding[k2], 1))
    for line in open('data/cl2_pairs_test.txt'):
        parts = line.split('|')
        k1, k2, w = int(parts[0]), int(parts[1]), float(parts[2])
        testing_ws.add(k1)
        testing_ws.add(k2)
        if w < 0:
            k1, k2 = k2, k1
        ps_ts.append((embedding[k1] + embedding[k2], 1))
    for line in open('data/cl2_no_relation_training.txt'):
        parts = line.split('|')
        k1, k2 = int(parts[0]), int(parts[1])
        training_ws.add(k1)
        training_ws.add(k2)
        ns_tr.append((embedding[k1] + embedding[k2], 0))
    for line in open('data/cl2_no_relation_test.txt'):
        parts = line.split('|')
        k1, k2 = int(parts[0]), int(parts[1])
        testing_ws.add(k1)
        testing_ws.add(k2)
        ns_ts.append((embedding[k1] + embedding[k2], 0))

    ps_tr = ps_tr * 3
    ps_ts = ps_ts * 3

    train = ps_tr + ns_tr
    test = ps_ts + ns_ts

    shuffle(train)
    shuffle(test)

    print('# of training positives: ', len(ps_tr))
    print('# of training negatives: ', len(ns_tr))
    print('# of testing positives: ', len(ps_ts))
    print('# of testing positives: ', len(ns_ts))
    print('# of training & testing word intersection: ', len(training_ws & testing_ws))

    X_train, y_train = split(train)
    X_test, y_test = split(test)
    return X_train, X_test, y_train, y_test

    # ps = ps * 7
    # print('# of positives: ', len(ps))
    # print('# of negatives: ', len(ns))
    # points = ps + ns
    # shuffle(points)
    #
    # data = [i[0] for i in points]
    # target = [i[1] for i in points]
    # print('Shuffling and split: test_percent: {}'.format(test_percent))
    # return train_test_split(data, target, test_size=test_percent, random_state=0)


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


if __name__ == '__main__':
    read_embedding()
    X_train, X_test, y_train, y_test = read_data()

    model = Sequential()
    model.add(Dense(64, input_dim=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', precision, recall, fmeasure])

    model.fit(X_train, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(X_test, y_test, batch_size=128)
