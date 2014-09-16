#Code to Evaluate the methods.

import math
import numpy as np
import nltk
import sys
import json
from pprint import pprint
from collections import Counter
import pdb
import GenerateTopicModel as GTM


clean=lambda orig: "".join([x for x in orig if ord(x) < 128])

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
"""
class Eval:
    def __init__(self):
"""

def getKLScore(summModelCopy, pModelCopy):

    for word in pModelCopy.keys():
        if not(summModelCopy.has_key(word)):
            summModelCopy[word]=0.001 # this will add an offset. Will be normalized next

    for word in summModelCopy.keys():
        if not(pModelCopy.has_key(word)):
            pModelCopy[word]=0.001 # this will add an offset. Will be normalized next

    #build index and create a list
    t=0; indexWords={}
    allWords=pModelCopy.keys()
    allWords.sort()
    pMlist=[];sMlist=[]
    for word in allWords:
        indexWords[word]=t
        pMlist.append(pModelCopy[word])
        sMlist.append(summModelCopy[word])
        t+=1

    # Normalise the pdf to make to sum upto one
    #
    pdel=(sum(pMlist)-1)/len(pMlist);
    pMlistMod= [x-pdel for x in pMlist];
    sdel=(sum(sMlist)-1)/len(sMlist);
    sMlistMod= [x-sdel for x in sMlist];
    #pdb.set_trace()
    KLD=kl(pMlistMod,sMlistMod)

    return KLD

def getSentimentScoreAllText(textList):

    #Read the Sentiment Dictionary
    f = open('senti_words.txt','r')
    sentiDict={}
    for line in f:
        new = line.split('\t')
        #pdb.set_trace();
        if (len(new)>1):
            sentiDict[new[0].rstrip()]=int(new[1].rstrip())
        else:
            sentiDict[new[0].rstrip()]=0


    score=np.array([0.0 for x in range(len(textList))])
    textSentiDict={};

    for i in range(len(textList)):
        words = textList[i].split(' ')
        for word in words:
            try:
                score[i] += sentiDict[word]
            except KeyError:
                score[i] +=0
        textSentiDict[i]=score[i];

    return ( textSentiDict, sentiDict);


def getSentimentAll(textList, sentiDict):
    posScore = 0;
    negScore = 0;
    neutScore= 0;
    keyword='All'
    sentiValue = {}
    score=np.array([0.0 for x in range(len(textList))])
    for i in range(len(textList)):
        words = textList[i].split(' ')
        for word in words:
            try:
                score[i] += sentiDict[word]
            except KeyError:
                score[i] +=0
    for i in range(len(score)):
        if score[i]>0:
            posScore += 1
        elif score[i]<0:
            negScore += 1
        elif score[i]==0:
            neutScore +=1
    sentiValue['positive'] = posScore/float(len(textList))*100
    sentiValue['negative'] = negScore/float(len(textList))*100
    sentiValue['neutral'] = neutScore/float(len(textList))*100

    #Sentiment Score just based on positive and negative sentiment.
    sentiValue['positive'] = posScore/float(posScore+negScore)*100
    sentiValue['negative'] = negScore/float(posScore+negScore)*100
    return sentiValue

def getSentimentKeywords(textList,keyword, sentiDict):
    score = 0
    positive={}
    negative={}
    neutral={}
    count={}
    textSenti = ['positive','negative','neutral']

    #sentiValue = [[0.0 for i in range(len(textSenti))] for j in range(len(keyword))]
    sentiValue=dict();

    for k in range(len(keyword)):
        count[keyword[k]]=0.0
        positive[keyword[k]]=0.0
        negative[keyword[k]]=0.0
        neutral[keyword[k]]=0.0

    for i in range(len(textList)):
        line =  textList[i].lower()
        for j in range(len(keyword)):
            score=0.0
            if keyword[j].lower() in line:
                wordKey=keyword[j].lower()
                count[wordKey]+=1
                words = line.split(' ')
                for word in words:
                    try:
                        score = score+sentiDict[word]
                    except KeyError:
                        score =score+0

                if (score>0):
                    positive[wordKey]+=1
                elif (score<0):
                    negative[wordKey]+=1
                elif (score==0):
                    neutral[wordKey]+=1


    for i in range(len(keyword)):
        neutral[keyword[i]] = count[keyword[i]]-positive[keyword[i]]-negative[keyword[i]]

    for word in keyword:
        den=positive[word]+negative[word]
        if (den!=0):
            sentiValue[word]=[positive[word]/float(den)*100, negative[word]/float(den)*100]
        else:
            sentiValue[word]=[0.0,0.0]


    return (sentiValue, count)



def EvalSentiment(summText, keyword):
    f = open('senti_words.txt','r')
    sentiDict={}
    for line in f:
        new = line.split('\t')
        #pdb.set_trace();
        if (len(new)>1):
            sentiDict[new[0].rstrip()]=int(new[1].rstrip())
        else:
            sentiDict[new[0].rstrip()]=0

    score = {}
    sentiScore = {}
    count=0;
    if ((len(keyword)==1) & (keyword[0]=='All')):
        sentiScore = getSentimentAll(summText, sentiDict)
    else:
        [sentiScore, count] = getSentimentKeywords(summText, keyword, sentiDict)

    return (sentiScore, count)
