from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# Do some text preprocessing
def cleanText(corpus):
    corpus = [z.lower().replace('\n','').split() for z in corpus]
    return corpus

def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

if __name__ == '__main__':

    with open('/Users/zxy/Desktop/IRDM group/sentiment_analysis_python-master/data/positive.txt', 'r') as infile:
        pos_tweets = infile.readlines()

    with open('/Users/zxy/Desktop/IRDM group/sentiment_analysis_python-master/data/negative.txt', 'r') as infile:
        neg_tweets = infile.readlines()

    # use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.25)
    x_train = cleanText(x_train)
    x_test = cleanText(x_test)

    n_dim = 300
    # Initialize model and build vocab
    w2v = Word2Vec(size=n_dim, min_count=10)
    w2v.build_vocab(x_train)
    # Train the model
    w2v.train(x_train)

    # move dataset into a gaussian distribution with a mean of zero
    # meaning that values above the mean will be positive, and those below the mean will be negative
    train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
    train_vecs = scale(train_vecs)

    # Train word2vec on test tweets
    w2v.train(x_test)

    # Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
    test_vecs = scale(test_vecs)

    # Use Stochastic Logistic Regression on training set
    # then assess model performance on test set
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)


    y_hat = lr.predict(test_vecs)
    p_and_rec = precision_recall_fscore_support(y_test, y_hat, beta = 1.0, labels = None, pos_label = 1, average = None, warn_for = ('precision', 'recall', 'f-score'), sample_weight = None)

    print('Test Accuracy: %.2f%%' %lr.score(test_vecs, y_test))
    print(p_and_rec)


