from pprint import pprint

def read_data(file):
    with open(file, 'r') as f:
        data = []
        for line in f.read().splitlines():
            data.append(line.split('\t'))
        data = data[1:] # Header절 건너뛰기 id	document	label
    return data

test_data = read_data('ratings_test.txt')
train_data = read_data('ratings_train.txt')

# print(len(test_data))
# print(len(test_data[0]))

# print(len(train_data))
# print(len(train_data[0]))

'''
# Since it takes long time to process this everytime we run, save train and test tokens using joblib

from konlpy.tag import Okt

twit = Okt()

def tokenize(doc):
    return ['/'.join(t) for t in twit.pos(doc, norm=True, stem=True)]

test_doc = []
train_doc = []
for row in test_data:
    test_doc.append((tokenize(row[1]),row[2]))
for row in train_data:
    train_doc.append((tokenize(row[1]),row[2]))

import joblib

# joblib.dump(test_doc, 'test_token.pkl')
# joblib.dump(train_doc, 'train_token.pkl')
'''
import joblib
test_doc = joblib.load('test_token.pkl')
train_doc = joblib.load('train_token.pkl')

tokens = []
for d in train_doc:
    for t in d[0]:
        tokens.append(t)
# print(len(tokens))


import nltk

#testing nltk

text = nltk.Text(tokens, name='NMSC')
'''
print(len(text.tokens))
print(len(set(text.tokens)))
pprint(text.vocab().most_common(10))

#since the text.plot() is printing square instead of most_common words set font

from matplotlib import font_manager, rc
font_fname = '/Library/Fonts/Arial Unicode.ttf/'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

text.plot(50)


# words = []
# for i in text.vocab().most_common(2000):
#     words.append(i[0])

# def term_exists(doc):
#     term = {}
#     for word in words:
#         term['exists({})'.format(word)] = (word in set(doc))
#     return term

# train_xy = []
# for d, c in train_doc:
#     train_xy.append((term_exists(d),c))

# test_xy = []
# for d, c in test_doc:
#     test_xy.append((term_exists(d), c))

# joblib.dump(train_xy, 'train_xy.pkl')
# joblib.dump(test_xy, 'test_xy.pkl')

train_xy = joblib.load('train_xy.pkl')
test_xy = joblib.load('test_xy.pkl')

classifier = nltk.NaiveBayesClassifier.train(train_xy)
print(nltk.classify.accuracy(classifier, test_xy))

classifier.show_most_informative_features(10)


'''

from collections import namedtuple
TaggedDocument = namedtuple('TaggedDocument', 'words tags')
tagged_train_docs = []
for d, c in train_doc:
  tagged_train_docs.append(TaggedDocument(d,[c]))

tagged_test_docs = []
for d, c in test_doc:
  tagged_test_docs.append(TaggedDocument(d, [c]))

from gensim.models import doc2vec

doc_vectorizer = doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, seed=1234)
doc_vectorizer.build_vocab(tagged_train_docs)

for epoch in range(10):
    doc_vectorizer.train(tagged_train_docs, total_examples=doc_vectorizer.corpus_count, epochs=doc_vectorizer.epochs)
    doc_vectorizer.alpha -= 0.002
    doc_vectorizer.min_alpha = doc_vectorizer.alpha

train_x = []
for doc in tagged_train_docs:
    train_x.append(doc_vectorizer.infer_vector(doc.words))
train_y = []
for doc in tagged_train_docs:
    train_y.append(doc.tags[0])
test_x = []
for doc in tagged_test_docs:
    test_x.append(doc_vectorizer.infer_vector(doc.words))

test_y = []
for doc in tagged_test_docs:
    test_y.append(doc.tags[0])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=1234)
classifier.fit(train_x, train_y)
print(classifier.score(test_x, test_y))
