import re
import nltk
import math
import collections
import itertools

from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from svmutil import *


def process(tweet):
    # Convert to lower case, useful!!!
    tweet = tweet.rstrip().lower()
    # Convert www.* or https?://* to URL, not useful!!!
    #tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # Convert @username to AT_USER, useful!
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    # Remove additional white spaces,same!!!
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word, not useful!!!
    #tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # trim, useful!!!
    tweet = tweet.strip('\'"')
    tweet = re.findall(r"[\w']+|[.,!?;]", tweet)
    return tweet

def create_word_scores():
    # creates lists of all positive and negative words
    posWords = []
    negWords = []
    with open('/Users/Krystal/Desktop/sentiment_analysis_python-master/data/positive.txt', 'r') as posSentences:
        for i in posSentences:
            posWords.append(process(i))
    with open('/Users/Krystal/Desktop/sentiment_analysis_python-master/data/negative.txt', 'r') as negSentences:
        for i in negSentences:
            negWords.append(process(i))
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    # build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        # replace two or more with two occurrences,same!!!
        word = replaceTwoOrMore(word)
        # strip punctuation
        word = word.strip('\'"?,.')
        # check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in negWords:
        # replace two or more with two occurrences,same!!!
        word = replaceTwoOrMore(word)
        # strip punctuation
        word = word.strip('\'"?,.')
        # check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1
    # finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    # builds dictionary of word scores based on chi-squared test
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    return word_scores

def replaceTwoOrMore(w):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", w)

def find_best_words(word_scores, number):
     best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
     best_words = set([w for w, s in best_vals])
     return best_words

def getFeature(tweet):
    featureVector = dict()
    for word in tweet:
        if word in best_words:
            # replace two or more with two occurrences, useful!!!
            word = replaceTwoOrMore(word)
            # strip punctuation, not useful!!! do not use.
            #word = word.strip('\'"?,.')
            # check if the word stats with an alphabet, same!!!
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
            featureVector[word] = True
    return featureVector


def naiveBayesClassifier(trainSet, testSet):
    referenceSets = collections.defaultdict(set)
    predictedSets = collections.defaultdict(set)

    NBClassifier = nltk.NaiveBayesClassifier.train(trainSet)

    for i, (features, label) in enumerate(testSet):
        referenceSets[label].add(i)
        predicted = NBClassifier.classify(features)
        predictedSets[predicted].add(i)

    # prints metrics to show how well the feature selection did
    print 'train on %d instances, test on %d instances' % (len(trainSet), len(testSet))
    print 'accuracy:', nltk.classify.util.accuracy(NBClassifier, testSet)
    #print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], predictedSets['pos'])
    #print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], predictedSets['pos'])
    #print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], predictedSets['neg'])
    #print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], predictedSets['neg'])
    #print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], predictedSets['neg'])
    NBClassifier.show_most_informative_features(10)


def maxEntropyClassifier(trainSet, testSet):
    referenceSets = collections.defaultdict(set)
    predictedSets = collections.defaultdict(set)

    MEClassifier = nltk.MaxentClassifier.train(trainSet, max_iter=5)

    for i, (features, label) in enumerate(testSet):
        referenceSets[label].add(i)
        predicted = MEClassifier.classify(features)
        predictedSets[predicted].add(i)

    # prints metrics to show how well the feature selection did
    print 'train on %d instances, test on %d instances' % (len(trainSet), len(testSet))
    print 'accuracy:', nltk.classify.util.accuracy(MEClassifier, testSet)
    print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], predictedSets['pos'])
    print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], predictedSets['pos'])
    print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], predictedSets['neg'])
    print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], predictedSets['neg'])
    MEClassifier.show_most_informative_features(10)


if __name__ == '__main__':

    # open file
    posFile = open('/Users/Krystal/Desktop/sentiment_analysis_python-master/data/positive.txt', 'r')
    negFile = open('/Users/Krystal/Desktop/sentiment_analysis_python-master/data/negative.txt', 'r')
    testFile = open('/Users/Krystal/Desktop/sentiment_analysis_python-master/data/Twitter_test_json_new.txt','r')
    #testFile_date = open('/Users/Krystal/Desktop/sentiment_analysis_python-master/data/Twitter_test_json_date.txt','r')
    #stopWords = getStopWordList('/Users/Krystal/Desktop/sentiment_analysis_python-master/data//stopWords.txt')

    # find best words
    word_scores = create_word_scores()
    best_words = find_best_words(word_scores, 50000)

    # extract features
    posFeatures = []
    negFeatures = []
    for posTweet in posFile:
        posTweet = process(posTweet)
        posWords = [getFeature(posTweet), 'pos']
        posFeatures.append(posWords)

    for negTweet in negFile:
        negTweet = process(negTweet)
        negWords = [getFeature(negTweet), 'neg']
        negFeatures.append(negWords)

    # selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures) * 3 / 4))
    negCutoff = int(math.floor(len(negFeatures) * 3 / 4))
    trainSet = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testSet = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    # Naive Bayes classifier
    #naiveBayesClassifier(trainSet, testSet)

    # Max Entropy Classifier
    #maxEntropyClassifier(trainSet, testSet)

    # Test the classifier
    # testTweet = 'i feel sad today'
    # processedTestTweet = process(testTweet)
    # print NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))

    # Test NB classifier
    #NBClassifier = nltk.NaiveBayesClassifier.train(trainSet)
    #for tweet in testFile:
        #content = tweet.split("\t")
        #processedTestTweet = process(content[1])
        #date = content[0]
        #result = NBClassifier.classify(getFeature(processedTestTweet))
        #print result
        #with open('/Users/Krystal/Desktop/sentiment_analysis_python-master/data/NB_output_new','a') as f:
            #f.write('%r\t%r\n' % (date,result))

    # Test ME classifier

    MEClassifier = nltk.MaxentClassifier.train(trainSet, max_iter=5)
    for tweet in testFile:
        content = tweet.split("\t")
        processedTestTweet = process(content[1])
        date = content[0]
        result = MEClassifier.classify(getFeature(processedTestTweet))
        print result
        with open('/Users/Krystal/Desktop/sentiment_analysis_python-master/data/ME_output_new','a') as f:
            f.write('%r\t%r\n' % (date,result))





































# import re
# import nltk
# import math
# import collections
# import itertools
#
# from nltk.metrics import BigramAssocMeasures
# from nltk.probability import FreqDist, ConditionalFreqDist
# from svmutil import *
#
#
# def process(tweet):
#     # Convert to lower case
#     tweet = tweet.rstrip().lower()
#     # Convert www.* or https?://* to URL
#     tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
#     # Convert @username to AT_USER
#     tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
#     # Remove additional white spaces
#     tweet = re.sub('[\s]+', ' ', tweet)
#     # Replace #word with word
#     tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
#     # trim
#     tweet = tweet.strip('\'"')
#
#     tweet = re.findall(r"[\w']+|[.,!?;]", tweet)
#     return tweet
#
# def getStopWordList(stopWordListFileName):
#     # read the stopwords file and build a list
#     stopWords = []
#     stopWords.append('AT_USER')
#     stopWords.append('URL')
#
#     fp = open(stopWordListFileName, 'r')
#     line = fp.readline()
#     while line:
#         word = line.strip()
#         stopWords.append(word)
#         line = fp.readline()
#     fp.close()
#     return stopWords
#
#
# def create_word_scores():
#     # creates lists of all positive and negative words
#     posWords = []
#     negWords = []
#     with open('/Users/zxy/Desktop/IRDM group/sentiment_analysis_python-master/data/positive.txt', 'r') as posSentences:
#         for i in posSentences:
#             posWords.append(process(i))
#     with open('/Users/zxy/Desktop/IRDM group/sentiment_analysis_python-master/data/negative.txt', 'r') as negSentences:
#         for i in negSentences:
#             negWords.append(process(i))
#     posWords = list(itertools.chain(*posWords))
#     negWords = list(itertools.chain(*negWords))
#
#     # build frequency distibution of all words and then frequency distributions of words within positive and negative labels
#     word_fd = FreqDist()
#     cond_word_fd = ConditionalFreqDist()
#     for word in posWords:
#         # replace two or more with two occurrences
#         word = replaceTwoOrMore(word)
#         # strip punctuation
#         word = word.strip('\'"?,.')
#         # check if the word stats with an alphabet
#         val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
#         # ignore if it is a stop word
#         if (word in stopWords or val is None):
#             continue
#         else:
#             word_fd[word] += 1
#             cond_word_fd['pos'][word] += 1
#     for word in negWords:
#         # replace two or more with two occurrences
#         word = replaceTwoOrMore(word)
#         # strip punctuation
#         word = word.strip('\'"?,.')
#         # check if the word stats with an alphabet
#         val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
#         # ignore if it is a stop word
#         if (word in stopWords or val is None):
#             continue
#         else:
#             word_fd[word] += 1
#             cond_word_fd['neg'][word] += 1
#     # finds the number of positive and negative words, as well as the total number of words
#     pos_word_count = cond_word_fd['pos'].N()
#     neg_word_count = cond_word_fd['neg'].N()
#     total_word_count = pos_word_count + neg_word_count
#
#     # builds dictionary of word scores based on chi-squared test
#     word_scores = {}
#     for word, freq in word_fd.iteritems():
#         pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
#         neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
#         word_scores[word] = pos_score + neg_score
#     return word_scores
#
#
# def find_best_words(word_scores, number):
#     best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
#     best_words = set([w for w, s in best_vals])
#     return best_words
#
#
# def replaceTwoOrMore(w):
#     # look for 2 or more repetitions of character and replace with the character itself
#     pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
#     return pattern.sub(r"\1\1", w)
#
#
# def getFeature(best_words, tweet):
#     featureVector = dict()
#     for word in tweet:
#         if word in best_words:
#             # replace two or more with two occurrences
#             word = replaceTwoOrMore(word)
#             # strip punctuation
#             word = word.strip('\'"?,.')
#             # check if the word stats with an alphabet
#             val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
#             # ignore if it is a stop word
#             if (word in stopWords or val is None):
#                 continue
#             else:
#                 featureVector[word] = True
#     return featureVector
#
#
# def naiveBayesClassifier(trainSet, testSet):
#     referenceSets = collections.defaultdict(set)
#     predictedSets = collections.defaultdict(set)
#
#     NBClassifier = nltk.NaiveBayesClassifier.train(trainSet)
#
#     for i, (features, label) in enumerate(testSet):
#         referenceSets[label].add(i)
#         predicted = NBClassifier.classify(features)
#         predictedSets[predicted].add(i)
#
#     # prints metrics to show how well the feature selection did
#     print 'train on %d instances, test on %d instances' % (len(trainSet), len(testSet))
#     print 'accuracy:', nltk.classify.util.accuracy(NBClassifier, testSet)
#     print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], predictedSets['pos'])
#     print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], predictedSets['pos'])
#     print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], predictedSets['neg'])
#     print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], predictedSets['neg'])
#     NBClassifier.show_most_informative_features(10)
#
#
# def maxEntropyClassifier(trainSet, testSet):
#     referenceSets = collections.defaultdict(set)
#     predictedSets = collections.defaultdict(set)
#
#     MEClassifier = nltk.MaxentClassifier.train(trainSet)
#
#     for i, (features, label) in enumerate(testSet):
#         referenceSets[label].add(i)
#         predicted = MEClassifier.classify(features)
#         predictedSets[predicted].add(i)
#
#     # prints metrics to show how well the feature selection did
#     print 'train on %d instances, test on %d instances' % (len(trainSet), len(testSet))
#     print 'accuracy:', nltk.classify.util.accuracy(MEClassifier, testSet)
#     print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], predictedSets['pos'])
#     print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], predictedSets['pos'])
#     print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], predictedSets['neg'])
#     print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], predictedSets['neg'])
#     MEClassifier.show_most_informative_features(10)
#
#
# if __name__ == '__main__':
#
#     # open file
#     posFile = open('/Users/zxy/Desktop/IRDM group/sentiment_analysis_python-master/data/positive.txt', 'r')
#     negFile = open('/Users/zxy/Desktop/IRDM group/sentiment_analysis_python-master/data/negative.txt', 'r')
#     stopWords = getStopWordList('/Users/zxy/Desktop/IRDM group/sentiment_analysis_python-master/data/stopWords.txt')
#
#     # find best words
#     word_scores = create_word_scores()
#     best_words = find_best_words(word_scores, 20000)
#
#     # extract features
#     posFeatures = []
#     negFeatures = []
#     for posTweet in posFile:
#         posTweet = process(posTweet)
#         posWords = [getFeature(best_words, posTweet), 'pos']
#         posFeatures.append(posWords)
#
#     for negTweet in negFile:
#         negTweet = process(negTweet)
#         negWords = [getFeature(best_words, negTweet), 'neg']
#         negFeatures.append(negWords)
#
#     print(len(posFeatures) + len(negFeatures))
#
#     # selects 3/4 of the features to be used for training and 1/4 to be used for testing
#     posCutoff = int(math.floor(len(posFeatures) * 3 / 4))
#     negCutoff = int(math.floor(len(negFeatures) * 3 / 4))
#     trainSet = posFeatures[:posCutoff] + negFeatures[:negCutoff]
#     testSet = posFeatures[posCutoff:] + negFeatures[negCutoff:]
#
#     # Naive Bayes classifier
#     naiveBayesClassifier(trainSet, testSet)
#
#     # Max Entropy Classifier
#     maxEntropyClassifier(trainSet, testSet)
#
#     # Support Vector Machines
#     # posFeatureSet = []
#     # for i, (features, label) in enumerate(posFeatures):
#     #     posFeatureSet.append(features)
#
#     # Test the classifier
#     # testTweet = 'i feel sad today'
#     # processedTestTweet = process(testTweet)
#     # print NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))



