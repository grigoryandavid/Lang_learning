import sklearn.feature_extraction.text

train_sent = []
with open('train.txt', 'r', encoding='UTF8') as inf:
    for i in range(0, 5000):
        sent = inf.readline()
        train_sent.append([sent[3:(len(sent) - 1)], sent[0:2]])

test_sent = []

LN = {'EN': 0, 'NL': 1, 'SV': 2, 'LT': 3, 'CS': 4, 'EE': 5, 'IT': 6, 'HU': 7, 'PL': 8, 'FI': 9, 'LV': 10, 'DA': 11,
      'DE': 12, 'ES': 13, 'FR': 14}
trainTexts = [p[0] for p in train_sent]
trainLN = [LN[p[1]] for p in train_sent]
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
#v = vectorizer.fit(trainTexts + test_sent)
v = vectorizer.fit(trainTexts)
trainVec = vectorizer.transform(trainTexts)
trainVec.toarray()

from sklearn.naive_bayes import GaussianNB
import numpy

clf = GaussianNB().fit(trainVec.toarray(), trainLN)
for k in range(0, 15):
    with open('test.txt', 'r', encoding='UTF8') as inf:
        for i in range(0, 15000):
            sent = inf.readline()
            if (1000 * k) <= i < (1000 * (k + 1)):
                test_sent.append(sent[0:(len(sent) - 1)])
        testVec = vectorizer.transform(test_sent).toarray()
        pred = clf.predict(testVec)
        LN_rev = {}
        for j in LN.keys():
            LN_rev[LN[j]] = j
        with open('output.txt', 'a', encoding='UTF8') as ouf:
            for j in range(0, len(pred)):
                ouf.write(LN_rev[pred[j]])
                ouf.write('\n')
    test_sent.clear()
    testVec = numpy.empty(testVec.shape)
    pred = numpy.empty(pred.shape)


