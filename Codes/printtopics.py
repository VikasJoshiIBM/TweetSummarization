import sys, os, re, random, math, urllib2, time, cPickle
import numpy

import onlineldavb

def printtopics1(i):
    """
    Displays topics fit by onlineldavb.py. The first column gives the
    (expected) most prominent words in the topics, the second column
    gives their (expected) relative prominence.
    """
    vocabfile = 'vocab7.txt'
    vocab = str.split(file(vocabfile).read())
    #testlambda = numpy.loadtxt(sys.argv[2])
    testlambda = numpy.loadtxt('lambda-15.dat')
    t={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
    fw = open('topics_result_'+str(i)+'.txt','w')
    for k in range(0, len(testlambda)):
        lambdak = list(testlambda[k, :])
        lambdak = lambdak / sum(lambdak)
        temp = zip(lambdak, range(0, len(lambdak)))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        key = lambda x: x[0]
        print 'topic %d:' %(k)
        fw.write("topic %d\n" % k)
        print (key)
        # feel free to change the "53" here to whatever fits your screen nicely.
        for i in range(0, 5):
            fw.write("%20s  \t---\t  %.10f\n" % (vocab[temp[i][1]], temp[i][0]))
            t[k].append(vocab[temp[i][1]])
            print '%20s  \t---\t  %.10f' % (vocab[temp[i][1]], temp[i][0])
        #t[k] = t[k].replace(',','-')
        t[k] = '-'.join(t[k])
        print t[k]
        print
    return t

#printtopics1('TRY')