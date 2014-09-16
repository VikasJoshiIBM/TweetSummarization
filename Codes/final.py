
from __future__ import unicode_literals
import math
import numpy
#import nltk
import sys
import json
from pprint import pprint
import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import unicodedata
import onlineldavb
#import wikirandom
import pdb

def is_ascii(s):
    return all(ord(c) < 128 for c in s)
'''
values = []
with open('tweets.txt') as json_file:
	for line in json_file:
        	#print (line);
		values.append( json.loads(line.decode('utf-8', 'ignore')) );
'''


def ldafinal(docset):
    # The number of documents to analyze each iteration
    batchsize = len(docset)
    # The total number of documents in Wikipedia
    D = len(docset)
    # The number of topics
    K =10

    # How many documents to look at

    documentstoanalyze = int(D/batchsize)


    # Our vocabulary
    vocab = file('./vocab7.txt').readlines()
    W = len(vocab)

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
    #print (olda)
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)
    #docset = list()

    #for i in range(len(values)):
    #   docset.append( values[i]['text'] )
    documentstoanalyze=20;
    for iteration in range(0, documentstoanalyze):

        (gamma, bound) = olda.update_lambda(docset)
        (wordids, wordcts) = onlineldavb.parse_doc_list(docset, olda._vocab)
        perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))

        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, olda._rhot, numpy.exp(-perwordbound))
        if (iteration % 5 == 0):
            numpy.savetxt('lambda-%d.dat' % iteration, olda._lambda)
            numpy.savetxt('gamma-%d.dat' % iteration, gamma)

