#from numpy import *
import math
import numpy
#import nltk
import sys
import json
from pprint import pprint

values = []
with open('data.json') as json_file:
	for line in json_file:
		values.append( json.loads(line) )
print(values[0]['created_at'])
print(values[1]['created_at'])
print(values[0]['text'])
print(values[1]['text'])


# def term_freq(text):
	# tf = {}
	# words=text.split(' ');
	# print (words)
	
	# for word in words:
		# #freq(word,text)
		# if word in tf:
			# tf[word]+=1
		# else:
			# tf[word]=1
	
	# print(tf)
	

def freq(word, doc):
	return doc.count(word)
 
 
def word_count(doc):
    return len(doc)
 
 
def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))
 
 
def num_docs_containing(word, values):
    count = 0
    for i in range(len(values)):
        if freq(word, values[i]['text']) > 0:
            count += 1
    return count
 
 
def idf(word, values):
    return math.log(len(values) /
            float(num_docs_containing(word, values)))
 
 
def tf_idf(word, doc, values):
    print((tf(word, doc) * idf(word, values)))

if __name__=="__main__":
	s = {}
	for i in range(len(values)):
		words=values[i]['text'].split(' '); 
		for word in words:
			if
			tf_idf(word,values[i]['text'],values)
	
	a = numpy.zeros(shape=(5,2))
	print(a[0][0])
# Matrix = [[0 for x in xrange(5)] for x in xrange(5)]
# Matrix[0][0] = 1
# Matrix[4][0] = 5

# print Matrix[0][0]
	# for d1 in range(6):
		# for d2 in range(6):
			# table[d1][d2]= d1+d2+2
	# print table
	
	#term_freq(values[i]['text'])
	#word_features(values[text],tf, true)


# def word_features(self,text,type='tf',normalize=True):
	# tf=self.term_freq(text)

	# if type=='tf':
		# features=tf
	# if type=='tfidf':
		# features={}
		# for word in tf.keys():
			# features[word]=tf[word]*self.idf[word]
	# if type=='ltc':
		# features={}
		# for word in tf.keys():
			# features[word]=(1+math.log(tf[word]))*self.idf[word]
	# if normalize:
		# norm=math.sqrt(sum([features[a]*features[a] for a in features]))
		# for a in features:
			# features[a]=features[a]*1.0/norm
	# return features
	# print(features)
