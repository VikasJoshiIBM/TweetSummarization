# Generate the Topic distributions from the tweet set
#v7: LexRank Summarization added into the code

#from numpy import *

import numpy as np
import math
import nltk
import sys
import json
import bow
from pprint import pprint
import Evaluation as Ev
import cPickle, string, numpy, getopt, sys, random, time, re, pprint, pdb
#import matplotlib
from pylab import *
import scipy
import UsefulFuncs
import Evaluation as Ev
import datetime as DT
import ner
import lexrank_summarization_tests as LR
import final
import printtopics
import onlineldavb
tagger = ner.SocketNER(host='localhost', port=8080)
#java -mx1000m -cp stanford-ner.jar edu.stanford.nlp.ie.NERServer -loadClassifier classifiers/english.muc.7class.distsim.crf.ser.gz -port 8080 -outputFormat inlineXML

clean=lambda orig: "".join([x for x in orig if ord(x) < 128])

"""
        if i not in bestTweetList:

            if (len(set(textWords).intersection(set(cleanWords))) == len(cleanWords)):
                j=0
                TFvec=np.array([0 for col in range(len(pModel.values()))])

                for wordVocab in cleanWords:
                    #print"i="+str(i)+"j="+str(j)
                    TFvec[j]= textWords.count(wordVocab)
               orig     j=j+1

                v2=TFvec[j];
                absv2=np.sqrt(np.dot(v2, v2));

                if ( not(UsefulFuncs.feq(absv1,0) | UsefulFuncs.feq(absv2,0)) ):
                    tweetScore=np.dot(v1, v2) / (absv1*absv2);
                else:
                    tweetScore=0.0
#                tweetScore[i]=tweetSimilarityScore(TFvec, fakeTweet);
                flagList.append(i);


    if (tweetScore.argmax()<TscoreLim):
        for i in range(len(textList)):
            if (i not in flagList):
                textWords=createStemmedWordList(textList[i])
                if (len(set(textWords).intersection(set(cleanWords))) == len(cleanWords)):
                    j=0
                    TFvec=np.array([0 for col in range(len(pModel.values()))])

                    for wordVocab in cleanWords:
                        print"i="+str(i)+"j="+str(j)
                        TFvec[j]= textWords.count(wordVocab)
                        j=j+1
                    tweetScore[i]=tweetSimilarityScore(TFvec, fakeTweet);
"""

#Named Entity Recognition:
def NER(textList):
    ner2={}
    nerVal = {}
    f = open('NER_final.txt','w')
    for i in range(len(textList)):
        line = textList[i]
        ner2 = tagger.get_entities(line)
        l = len(ner2)
        nerVal[i]=l*0.33
        f.write(str(i)+"     "+str(l)+"       "+str(ner2))
        f.write("\n")
    return (nerVal);

def NEReval_AllvsSumm(nerVal,summTweet):
    nerAll = 0
    nersumm = 0
    for i in range(len(nerVal)):
        nerAll = nerAll+nerVal[i]
    nerMRS = {}
    nerMRS = NER(summTweet)
    for i in range(len(nerMRS)):
        nersumm = nersumm+nerMRS[i]
    nerlist = []
    nerlist.append(nerAll)
    nerlist.append(nersumm)
    return nerlist


def veracity(values):
    follow = []
    retweet = []
    verified = []
    for i in range(len(values)):
        follow.append(values[i]['user']['followers_count'])
        verified.append(values[i]['user']['verified'])
        try:
            val = values[i]['retweeted_status']['retweet_count']
            retweet.append(val)
        except KeyError:
            #print "not retweeted"
            retweet.append('0')

    a1=1
    a2=1
    a3=1
    score = []
    ver = {}
    n = len(follow)

    for i in range(n):
        if verified=="true":
            v = 1;
        else:
            v = 0;
        f = 1-(math.exp(-follow[i]))
        r = 1-(math.exp(-int(retweet[i])))
        x = (a1*f+a2*r+a3*v)/(a1+a2+a3)
        ver[i] = x
        score.append(x)
            #score.append(follow[i] + int(retweet[i])+)
    return ver


def getModel(textList, Nterms):
    #Note few parameters are hard-coded. Can change if neccessary
    BOW=bow.BagOfWords();

    #for i in range(len(textList)):
    #print textList[0]
    BOW.addstoplist(stoplist)
    BOW.build_index(textList, max(min(len(textList)/50,50),1))
    BOW.est_probability(Nterms)
    return BOW
"""
    pdb.set_trace();

    outFile='./vocab.txt'
    f = open(outFile,'w')

    BOW.dump_vocab(f);
    #display BOW.build_index.vocab
    #print(vocab)
    #for i in range(len(text)):
    #    f.write("%s\n" % (BOW.vocab);
    f.close()
"""

def getSentimentProb(wordList, textList):
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
    sentimentVal=dict();sentimentProb=dict();sentiWordIndex=dict()
    for words in wordList:
            sentimentProb[words]=[0.0, 0.0]
            sentimentVal[words]=[0.0, 0.0]
            sentiWordIndex[words]=[]

    textSentiDict={};
    for i in range(len(textList)):
        words = textList[i].split(' ')
        for word in words:
            try:
                score[i] += sentiDict[word]
            except KeyError:
                score[i] +=0

        textSentiDict[i]=score[i]
        #Build the sentiment Model jointly wiht the word model
        #for intWords in set(wordList).intersection(set(words)):
        for intWords in wordList:
            if (intWords in textList[i]):
                senti=max(min(score[i],1.0),-1.0);
                if( (senti == 1.0) | (senti ==-1.0)):
                    sentimentVal[intWords] = [sentimentVal[intWords][0]+((1+senti)/float(2)), sentimentVal[intWords][1] + abs(((-1+senti)/float(2)))]
                    sentiWordIndex[intWords].append(i);

    for word in sentimentProb.keys():
        if( (sentimentVal[word][0]==0.0) & (sentimentVal[word][1]==0.0)):
            sentimentProb[word]=[0.5,0.5];
        else:
            sumVal=sum(sentimentVal[word]);
            sentimentProb[word]=[sentimentVal[word][0]/float(sumVal), sentimentVal[word][1]/float(sumVal)]

    return (sentimentProb, textSentiDict, sentiDict);


def getSentimentProb_FromEV(wordList, textList):

    sentimentVal=dict();sentimentProb=dict();sentiWordIndex=dict()
    for words in wordList:
            sentimentProb[words]=[0.0, 0.0]
            sentimentVal[words]=[0.0, 0.0]
            sentiWordIndex[words]=[]

    keyword=wordList;
    [sentimentVal, countAll]=Ev.EvalSentiment(textList,keyword);

    for i in range(len(sentimentVal)):
        sentimentProb[wordList[i]]=[sentimentVal[wordList[i]][0]/float(100.0),sentimentVal[wordList[i]][1]/float(100.0)];

    print sentimentProb;

    return sentimentProb;


def EvalKLD(summText, pModel):
    KLScore=[]
    #Get the model from the summarization text
    summModel=getModel(summText, len(pModel.values())).prob
    summModelCopy=summModel.copy()
    pModelCopy=pModel.copy()
    Niters=1
    KLScore.append(Ev.getKLScore(summModelCopy,pModelCopy)) #Niters is the number of iterations on which score is obtained

    return KLScore

def EvalKLDRandomSamp(textList,Niters,pModelCopy,Nsumm):

    KLScore=[];
    for iters in range(Niters):
        index=np.random.uniform(0,len(textList),Nsumm)
        indexALLRS=np.array([int(i) for i in index])
        summText=[textList[int(i)] for i in index]
        summModel=getModel(summText, len(pModelCopy.values())).prob
        summModelCopy=summModel.copy()
        pModelCopy=pModel.copy()
        Niters=1
        KLScore.append(Ev.getKLScore(summModelCopy,pModelCopy)) #Niters is the number of iterations on which score is obtained


        #KLScore.append(EvalKLD(summText, pModelCopy))

    return (KLScore, summModel, summText, indexALLRS);

def est_probability_vocab(textListsumm, vocab):
    sumDict=0; pModelEst={};
    cleanWords=cleanUpWords(vocab.keys())

    for word in cleanWords:
        pModelEst[word]=0.0

    for i in range(len(textListsumm)):
        tweetWordList=createStemmedWordList(textListsumm[i]);
        for word in cleanWords:
            pModelEst[word] = pModelEst[word] + tweetWordList.count(word);
            sumDict = sumDict + tweetWordList.count(word)

    for word in vocab:
        pModelEst[word] = pModelEst[word]/float(sumDict)

    if (not(UsefulFuncs.feq(sum(pModelEst.values()),1.0))):
        raise Exception('Probability Error: pdf doesnt sum to one')

    return pModelEst;

def createStemmedWordList(textline):
    wordList=[];
    stemmer=nltk.stem.PorterStemmer();
    words=nltk.word_tokenize(textline)
    wordList = [stemmer.stem(clean(word.lower().rstrip()))  for word in words]

    return wordList;

def getlengthpdf(textList, wordList):
    lenStat=[]; lenPdf={};
    for i in range(len(textList)):
        tweetWordList=createStemmedWordList(textList[i]);
        lenStat.append(len(set(tweetWordList).intersection(set(wordList))))
    lenPdf['mean']=mean(lenStat)
    lenPdf['stddev']=std(lenStat)

    return lenPdf

def dump_model(pModel,fid):
    for word in pModel.keys():
        print >>fid,"%s\t%f" % (word,pModel[word])



def buildpdfList(pModel):
    t=0; indexWords={}; pMlist=[]
    allWords=pModel.keys()
    for word in allWords:
        indexWords[word]=t
        pMlist.append(pModel[word])
        t+=1
    return pMlist,indexWords;

def tweetSimilarityScore(TFvec, fakeTweet):
    v1=TFvec; v2=fakeTweet;
    absv1=np.sqrt(np.dot(v1, v1));
    absv2=np.sqrt(np.dot(v2, v2));

    if ( not(UsefulFuncs.feq(absv1,0) | UsefulFuncs.feq(absv2,0)) ):
        tweetScore=np.dot(v1, v2) / (absv1*absv2);
    else:
        tweetScore=0.0

    return tweetScore

def cleanUpWords(wordList):
    cleanUpWords=[]
    stemmer=nltk.stem.PorterStemmer();
    cleanWords = [stemmer.stem(clean(word.lower().rstrip()))  for word in wordList]

    return cleanWords

def getSummTweet(fakeTweet, textList, pModel,bestTweetList, fid):


    tweetScore=np.array([-10.0 for col in range(len(textList))])
    cleanWords=cleanUpWords(pModel.keys())
    flagList=[];
    finalscore = []
    TscoreLim=1
    cnt=0
    w1=0.7; w2=0.3;
    minCS_score=0.7;

    f = open('senti_words.txt','r')
    sentiDict={}
    for line in f:
        new = line.split('\t')
        #pdb.set_trace();
        if (len(new)>1):
            sentiDict[new[0].rstrip()]=int(new[1].rstrip())
        else:
            sentiDict[new[0].rstrip()]=0

    v1=fakeTweet; absv1=np.sqrt(np.dot(v1, v1));
    score=np.array([0.0 for x in range(len(textList))])
    finalscore=np.array([0.0 for x in range(len(textList))])
    for i in range(len(textList)):
        if (i not in bestTweetList):
            textWords=createStemmedWordList(textList[i])
            j=0
            TFvec=np.array([0 for col in range(len(pModel.values()))])
            for wordVocab in cleanWords:
                #print"i="+str(i)+"j="+str(j)
                TFvec[j]= textWords.count(wordVocab)
                j=j+1

            v2=TFvec;
            absv2=np.sqrt(np.dot(v2, v2));
            den=absv1*absv2

            if ( not(UsefulFuncs.feq(den,0) ) ):
                #tweetScore[i]=float(np.dot(v1, v2)) / float(den);
                tweetScore[i]=np.divide(float(np.dot(v1, v2)), den)
            else:
                 tweetScore[i]=0.0
            #print >>fid, "%s\n%s\n%f\t%f\t%f\t%f\n\n" % (str(v1), str(v2), np.dot(v1, v2),den, tweetScore[i], sen)

            #f1=open('temp/wordVector'+str(i)+'.txt','w');
            #print >>fid, "%s\n%s\n%s\n%s\n%f\n\n" % (str(textWords), str(cleanWords),str(TFvec), str(fakeTweet), tweetScore[i])

            #f1.close()
            if (tweetScore[i]>minCS_score):
                words = textList[i].split(' ')
                for word in words:
                    try:
                        score[i] += sentiDict[word]
                    except KeyError:
                        score[i] +=0
                score[i]=min(abs(score[i]),3)/3;

            finalscore[i] = w1*tweetScore[i] + w2*score[i]
    #bestTweetScore=tweetScore.index(max(tweetScore))
    bestTweetIndex=finalscore.argmax()
    print "Found the best Tweet"
    #Dump the different variables for debugging!
    return (textList[bestTweetIndex], bestTweetIndex, tweetScore[bestTweetIndex], score[bestTweetIndex],finalscore[bestTweetIndex])

def dump_textList(textList,outFileName):
    fid=open(outFileName,'w');
    for i in range(len(textList)):
        y=clean(textList[i])
        print >>fid, "%s" % (clean(textList[i].rstrip().replace('\n','')));
    fid.close()



def getSummTweet_vSnt(fakeTweet, textList, pModel,bestTweetList, fid, textSentiDict, sentimentProb):

    tweetScore=np.array([-10.0 for col in range(len(textList))])
    cleanWords=cleanUpWords(pModel.keys())
    flagList=[];
    finalscore = []
    TscoreLim=1
    cnt=0
    w1=0.5; w2=0.5;
    minCS_score=0.8;

    #Prepocessing and initial calculations
    v1=fakeTweet; absv1=np.sqrt(np.dot(v1, v1));
    score=np.array([0.0 for x in range(len(textList))])
    finalscore=np.array([0.0 for x in range(len(textList))])

    #Create wordIndex List.
    count=0; wordIndexList=dict()
    for word in pModel.keys():
        wordIndexList[count]=word;
        count = count+1;

    sentimentIndex=dict()
    for i in range(len(textList)):
        if (i not in bestTweetList):
            textWords=createStemmedWordList(textList[i])
            j=0
            TFvec=np.array([0 for col in range(len(pModel.values()))])
            for wordVocab in cleanWords:
                #print"i="+str(i)+"j="+str(j)
                TFvec[j]= textWords.count(wordVocab)
                j=j+1

            v2=TFvec;
            absv2=np.sqrt(np.dot(v2, v2));
            den=absv1*absv2

            if ( not(UsefulFuncs.feq(den,0) ) ):
                #tweetScore[i]=float(np.dot(v1, v2)) / float(den);
                tweetScore[i]=np.divide(float(np.dot(v1, v2)), den)
            else:
                 tweetScore[i]=0.0
            #print >>fid, "%s\n%s\n%f\t%f\t%f\t%f\n\n" % (str(v1), str(v2), np.dot(v1, v2),den, tweetScore[i], sen)

            #f1=open('temp/wordVector'+str(i)+'.txt','w');


            #f1.close()

            #if (tweetScore[i]>minCS_score):
                #Calculate the sentiment score from the faketweet

    if (max(tweetScore) >minCS_score):
        sentiment=0.0
        indList=[];
        count=0;
        for fakewords in fakeTweet:
            if (fakewords!=0):
                for FWsingle in range(fakewords):
                    ind=numpy.random.multinomial(1,sentimentProb[wordIndexList[count]],1);
                    #print "word="+str(wordIndexList[count])+" SentimentProb="
                    #print sentimentProb[wordIndexList[count]]
                    indList.append(ind[0]);
            count=count+1;

        xarray=numpy.array(indList)
        if ((xarray.transpose()[0].sum()) > (xarray.transpose()[1].sum())):
            sentiment=1.0
        else:
            sentiment=-1.0

        for j in range(len(textList)):
            if(textSentiDict[j]*sentiment >= 1.0):
                #score[i]=min(abs(textSentiDict[i]),3)/3;
                score[j]=1.0
                finalscore[j] = w1*tweetScore[j] + w2*score[j]


        bestTweetIndex=finalscore.argmax()
        #print "Found the best Tweet"
        #print >>fid, "%s\n%s\n%f\t%f\t%f\t%f\n\n" % (str(textList[bestTweetIndex]), str(fakeTweet), tweetScore[bestTweetIndex], sentiment, score[i], finalscore[i])
        #bestTweetScore=tweetScore.index(max(tweetScore))


        #Dump the different variables for debugging!
    return (textList[bestTweetIndex], bestTweetIndex, tweetScore[bestTweetIndex], score[bestTweetIndex],finalscore[bestTweetIndex])


def getSummMRS(textList, Nsumm, pModel, lenPdf, textSentiDict, TFMat, parameters, scorener, scorevar):

    indexAll=[]
    cleanWords=cleanUpWords(pModel.keys())
    f = open('senti_words.txt','r')
    sentiDict={}
    for line in f:
        new = line.split('\t')
        #pdb.set_trace();
        if (len(new)>1):
            sentiDict[new[0].rstrip()]=int(new[1].rstrip())
        else:
            sentiDict[new[0].rstrip()]=0

    w1=parameters['w1']; w2=parameters['w2'];w3=parameters['w3'];w4=parameters['w4']
    minCS_score=parameters['minCS_score'];minFS_score=parameters['minFS_score'];
    flagList=[]; finalscore = []; bestTweetList=[];
    TscoreLim=1
    cnt=0
    sentimentIndex=dict();
    indexAllSelect=[]
    sentimentHistogram=dict();summTweetMRS=[]
    sentimentList=dict(); bestTweetScore=[]; bestSentimentScore=[]; sentimentEval=[]; bestFinalScore=[]

    for word in pModel.keys():
        sentimentHistogram[word]=[0.0, 0.0]

    #Create wordIndex List.
    count=0; wordIndexList=dict()
    for word in pModel.keys():
        wordIndexList[count]=word;
        count = count+1;

    fid=open('temp/SummParam'+'.txt','w');
    print >>fid, "Start of file\n";
    fid.close();
    fid=open('temp/SummParam'+'.txt','a');

    #Generate the length samples
    lenSamp=np.random.normal(lenPdf['mean'],lenPdf['stddev'],Nsumm)

    while (cnt<Nsumm):
        print "Summary tweet number:"+str(cnt)+"out of "+str(Nsumm)+"Tweets"
        x=int(ceil(max((lenSamp[cnt],1))))
        index=numpy.random.multinomial(x,pMlist,1)
        fakeTweet=index[0];
        #[Tweet, TweetIndex, tweetScore, SentiScore,finalscore]=getSummTweet_vSnt(index[0], textList, pModel, bestTweetList, fid, textSentiDict, sentimentProb)

        tweetScore=np.array([-10.0 for col in range(len(textList))])
        v1=np.array(fakeTweet); absv1=np.sqrt(np.dot(v1, v1));
        score=np.array([0.0 for x in range(len(textList))])
        finalscore=np.array([0.0 for x in range(len(textList))])
        indexMatchList=[];
        for i in range(len(textList)):
            if (i not in bestTweetList):
                TFvec=TFMat[i]
                v2=TFvec;
                absv2=np.sqrt(np.dot(v2, v2));
                den=absv1*absv2

                if ( not(UsefulFuncs.feq(den,0) ) ):
                    #tweetScore[i]=float(np.dot(v1, v2)) / float(den);
                    tweetScore[i]=np.divide(float(np.dot(v1, v2)), den)
                else:
                     tweetScore[i]=0.0

                if(tweetScore[i]>minCS_score):
                    indexMatchList.append(i);

                #print >>fid, "%s\n%s\n%f\t%f\t%f\t%f\n\n" % (str(v1), str(v2), np.dot(v1, v2),den, tweetScore[i], sen)

                #f1=open('temp/wordVector'+str(i)+'.txt','w');
                #print >>fid, "%s\n%s\n%s\n%s\n%f\n\n" % (str(textWords), str(cleanWords),str(TFvec), str(fakeTweet), tweetScore[i])

                #f1.close()
        if (max(tweetScore) >minCS_score):
            sentiment=0.0
            indList=[];
            count=0;
            fakewordsList=[];
            for fakewords in fakeTweet:
                if (wordIndexList[count] in sentiWordsImp):
                    if (fakewords!=0):
                        for FWsingle in range(fakewords):
                            ind=numpy.random.multinomial(1,sentimentProb[wordIndexList[count]],1);
                            #print "word="+str(wordIndexList[count])+" SentimentProb="
                            #print sentimentProb[wordIndexList[count]]
                            indList.append(ind[0]);
                            sentimentHistogram[wordIndexList[count]]= [(sentimentHistogram[wordIndexList[count]][0]+ind[0][0]), (sentimentHistogram[wordIndexList[count]][1]+ind[0][1])]
                            fakewordsList.append(wordIndexList[count]);
                count=count+1;

            if( len(indList)==0 ):
                sentiment=0.0
            else:
                xarray=numpy.array(indList)
                if ((xarray.transpose()[0].sum()) > (xarray.transpose()[1].sum())):
                    sentiment=1.0
                else:
                    sentiment=-1.0

            #for j in range(len(textList)):
            for j in indexMatchList:
                if(textSentiDict[j]*sentiment >= 1.0):
                    #score[i]=min(abs(textSentiDict[i]),3)/3;
                    score[j]=1.0
                    finalscore[j] = w1*tweetScore[j] + w2*score[j]+w3*scorener[j]+w4*scorevar[j];

            if (max(finalscore) > minFS_score):
                bestTweetIndex=finalscore.argmax()
                #Store the selected paramters
                Tweet=textList[bestTweetIndex]
                summTweetMRS.append(Tweet)
                bestTweetList.append(bestTweetIndex)
                indexAllSelect.append(index)
                sentimentList[cnt]=xarray
                bestTweetScore.append(tweetScore[bestTweetIndex])
                bestSentimentScore.append(textSentiDict[bestTweetIndex])
                sentimentEval.append(sentiment)
                bestFinalScore.append(finalscore[bestTweetIndex])
                cnt=cnt+1;
                #Print the parameters into thte file
                print >>fid, "faketweet=%s\nGenerated Sentiment=%s\nGenerated Tweet=%s" % (str(fakewordsList), str(sentiment), str(clean(Tweet)));

        #Store all the generated fakeTweets
        indexAll.append(index)

    fid.close()
    return (summTweetMRS, bestTweetList, indexAllSelect, sentimentList, bestTweetScore, bestSentimentScore, sentimentEval, bestFinalScore, sentimentHistogram);

def readInputandVeracity(inpTweetFile):
    maxLines=1000;
    values = []
    nlines=0;
    with open(inpTweetFile) as json_file:
        for x in xrange(maxLines):
            values.append( json.loads(json_file.next()) )
            #nlines=nlines+1;
        #for line in json_file:
        #    values.append( json.loads(line) )
    textList=[]
    for i in range(len(values)):
        textList.append(clean(values[i]["text"]))

    scorevar = {}
    scorevar = veracity(values)

    return (textList, scorevar)

def perplexity(pModel, summTextper):

    likelihood=np.array([0.0 for x in range(len(summTextper))]);
    TFVec=np.array([0.0 for x in range(len(pModel))]);
    for i in range(len(summTextper)):
        textWords=createStemmedWordList(summTextper[i])
        j=0
        lkHood=0;
        prob=1;
        for wordVocab in pModel.keys():
            #print"i="+str(i)+"j="+str(j)
            TFVec[j]= textWords.count(wordVocab)

            if (TFVec[j]!=0):
                prob=prob*(pModel[wordVocab])*TFVec[j]
            else:
                prob=prob*0.01;
            lkHood=math.log(prob);
            j=j+1

        #likelihood[i]=math.exp(-lkHood/len(summTextper))
        likelihood[i]=lkHood
    perScore=math.exp(-(sum(likelihood)/len(likelihood)))
    return perScore;

#def ModelRestorationSumm(textList, pModelCopy, Nsumm):
if __name__=="__main__":

    if (len(sys.argv) < 2):
        print "Expected 2 Arguemnts <input tweet file>, <output File to store Models>"
        exit(1);

    print "Reading the input File"
    inpTweetFile=sys.argv[1]; #Input Tweet File
    #inpTweetFile= 'TweetsMar26.txt'
    outVocabFile='vocab7.txt';

#--------------------------Defining Parameters-----------------------------------
    Nterms=10; #Number of terms to be chosen in the multinomial model
    Nsumm=30; #Number of tweets in summarized output
    w1=1.0; w2=0.0; w3=0.0; w4=0.0  # Weights for Topics, Sentiment, NER and Veracity
    sentiWordsImp=['bjp', 'congress', 'akbar', 'aap', 'modi', 'kejriw', 'namo'] #Important words around which the sentiment is restored. TODO: Make it generic enough to incorporate Nouns
    minCS_score=0.5;minFS_score=0.5;
    parameters=dict()
    parameters['w1']=w1;parameters['w2']=w2; parameters['w3']=w3; parameters['w4']=w4;
    parameters['sentiWordsImp']=sentiWordsImp
    parameters['minCS_score']=0.5; parameters['minFS_score']=0.5
    Niters=2; #Number of iterations of Summarization in order to compare the results
    LexRnk=1;
#----------------------------------------------------------------------------------



    [textList, scorevar] = readInputandVeracity(inpTweetFile)
    stoplist = []
    with open('StopWordList.txt', 'r') as StopWrds_file:
        for line in StopWrds_file:
            stoplist.append( line.rstrip() )


    #Dump File
    outFileName='temp/TextListAll.txt'
    #dump_textList(textList,outFileName);


    print "Building the vocabulary"
    BOW=bow.BagOfWords();
    BOW.addstoplist(stoplist);
    BOW.build_index(textList,max(min(len(textList)/50,200),1))
    f = open(outVocabFile,'w')
    BOW.dump_vocab(f)
#    TFMatrix=BOW.getTFMatrix(textList)
    f.close()

    #Build the Probability density Function
    print "Building the Model"
    pModel={}
    pModel=getModel(textList, Nterms).prob
    f=open('Model1.txt','w');
    dump_model(pModel,f)
    f.close()

    # get the length Model
    wordList=pModel.keys();
    lenPdf=getlengthpdf(textList, wordList)

    # Get the sentiment joint probability
    [textSentiDict, sentiDict] = Ev.getSentimentScoreAllText(textList);

    #[sentimentProb, textSentiDict, sentiDict]=getSentimentProb(pModel.keys(), textList)
    sentimentProb=getSentimentProb_FromEV(pModel.keys(), textList);


    #GEt the TF Matrix for the input Tweets
    cleanWords=cleanUpWords(pModel.keys())
    TFMat=[];
    for i in range(len(textList)):
        TFMat.append(np.array([0 for col in range(len(pModel.values()))]))

    TFvec=np.array([0 for col in range(len(pModel.values()))])
    for i in range(len(textList)):
        textWords=createStemmedWordList(textList[i])
        j=0
        for wordVocab in cleanWords:
            #print"i="+str(i)+"j="+str(j)
            TFMat[i][j]= textWords.count(wordVocab)
            j=j+1

    #Sampling using Model restoration approach
    summTweet=[];
    print "Summarizaing the tweets using Model Restoration Approach\n"
    print "Parameters: Nsumm="+str(Nsumm);
    [pMlist,indexWords]=buildpdfList(pModel)
    bestTweetList=[]
    outFileParam='temp/SummParameters.txt'
    f=open(outFileParam,'w')
    cnt=0;

    datetimeNow=DT.datetime.now();
    ResultsFile='../results/'+str(datetimeNow.date())+'_'+str(datetimeNow.hour)+'_'+str(datetimeNow.minute)+'_'+str(datetimeNow.second)+'.txt'
    fidres=open(ResultsFile,'w');

    MRSsentiScore=np.array([0.0 for x in range(Niters)]);
    RSsentiScore=np.array([0.0 for x in range(Niters)]);
    LRsentiScore=np.array([0.0 for x in range(Niters)]);
    MRSvsACT_std=np.array([0.0 for x in range(Niters)]);
    RSvsACT_std=np.array([0.0 for x in range(Niters)]);
    LRvsACT_std=np.array([0.0 for x in range(Niters)]);
    verscoreMRS=np.array([0.0 for x in range(Niters)]);
    verscoreRS=np.array([0.0 for x in range(Niters)]);
    verscoreLR=np.array([0.0 for x in range(Niters)]);
    NERscoreMRS=np.array([0.0 for x in range(Niters)]);
    NERscoreRS=np.array([0.0 for x in range(Niters)]);
    NERscoreLR=np.array([0.0 for x in range(Niters)]);
    nerMRS=np.array([0.0 for x in range(Niters)]);
    nerRS=np.array([0.0 for x in range(Niters)]);
    nerLR=np.array([0.0 for x in range(Niters)]);
    perMRS=np.array([0.0 for x in range(Niters)]);
    perRS=np.array([0.0 for x in range(Niters)]);
    perLR=np.array([0.0 for x in range(Niters)]);


    scorener = {}
    scorener = NER(textList)

    if(LexRnk==1):
        #LexRank Summarization
        #doc_dict={}
        #(doc_dict,test_documents) = LR.initialize(textList)
        #(res,summLexRank)=LR.test_cosine_matrix_creation(doc_dict, test_documents)
        (res,summLexRank)=LR.test_cosine_matrix_creation_v1(textList)

    '''[summTweetMRS, bestTweetList, indexAllSelect, sentimentList, bestTweetScore, bestSentimentScore, sentimentEval, bestFinalScore, sentimentHistogram] = getSummMRS(textList, Nsumm, pModel, lenPdf, textSentiDict, TFMat, parameters, scorener, scorevar);

    NitersRS=1;
    pModelCopy=[]; pModelCopy=pModel.copy();
    [KLscoreRS, sRSModel, textListsummRS, indexALLRS]=EvalKLDRandomSamp(textList, NitersRS, pModelCopy, Nsumm)
    '''
    #final.ldafinal(textList)
    #w = printtopics.printtopics1('Actual')


    for iter in range(Niters):

        #Get the summarization:
        [summTweetMRS, bestTweetList, indexAllSelect, sentimentList, bestTweetScore, bestSentimentScore, sentimentEval, bestFinalScore, sentimentHistogram] = getSummMRS(textList, Nsumm, pModel, lenPdf, textSentiDict, TFMat, parameters, scorener, scorevar);

        #pMRSModel=getModel(summTweet);
        pModelCopy=pModel.copy();
        pMRSModelvocab = est_probability_vocab(summTweetMRS, pModel)
        KLScoreMRSvocab=Ev.getKLScore(pMRSModelvocab, pModelCopy)

        outFileNameSumm='temp/textListMRSumm.txt'
        dump_textList(summTweetMRS, outFileNameSumm)
        print "Done with SUmmarization"

        #Evaluate the summarization
        KLScoreMRS=EvalKLD(summTweetMRS,pModel)

        #Evaluation of Random Sampling
        NitersRS=1;
        pModelCopy=[]; pModelCopy=pModel.copy();
        [KLscoreRS, sRSModel, textListsummRS, indexALLRS]=EvalKLDRandomSamp(textList, NitersRS, pModelCopy, Nsumm)
        sRSModelvocab = est_probability_vocab(textListsummRS, pModel)
        KLScoreRSvocab=Ev.getKLScore(sRSModelvocab, pModelCopy)
        pModelCopy=pModel.copy();
        sLRModelvocab = est_probability_vocab(summLexRank, pModel)
        KLScoreLRvocab=Ev.getKLScore(sLRModelvocab,pModelCopy);

        #print"KLScoreMRS="+KLScoreMRS[0]+" KLscorePS="+KLscorePS[0]
        sentimentProbGen={}
        for word in sentimentHistogram.keys():
            if( (sentimentHistogram[word][0]==0.0) & (sentimentHistogram[word][1]==0.0)):
                sentimentProbGen[word]=[0.5,0.5];
            else:
                sumVal=sum(sentimentHistogram[word]);
                sentimentProbGen[word]=[(sentimentHistogram[word][0]/float(sumVal)), (sentimentHistogram[word][1]/float(sumVal))];

        #Sentiment based Evaluation:
        #keyword=['bjp', 'congress', 'modi', 'rahul', 'gandhi', 'aap'];
        keyword=pModel.keys();
        [SentimentScoreALLText, countAll] = Ev.EvalSentiment(textList,keyword)
        [SentimentScoreMRS, countMRS] = Ev.EvalSentiment(summTweetMRS,keyword)
        [SentimentScoreRS, countRS] = Ev.EvalSentiment(textListsummRS,keyword)
        [SentimentScoreLR, countLR] = Ev.EvalSentiment(summLexRank,keyword)

        #print "LDA starts"
        #final.ldafinal(textList)
        #w = printtopics.printtopics1('Actual')
        #final.ldafinal(summTweetMRS)
        #x = printtopics.printtopics1('MRS')
        #final.ldafinal(textListsummRS)
        #y = printtopics.printtopics1('RS')
        #final.ldafinal(summLexRank)
        #z = printtopics.printtopics1('LR')
        #print "LDA done finally"
        #ft = open('topics.csv','w')
        #ft.write("Actual,MRS,RS,LR\n")
        #for i in range(0,9):
        #    ft.write("%s,%s,%s,%s\n" % (str(w[i]),str(x[i]),str(y[i]),str(z[i])))
        #ft.close()

        #Calculate the perplexity
        perMRS[iter]=perplexity(pModel, summTweetMRS)
        perRS[iter]=perplexity(pModel, textListsummRS)
        perLR[iter]=perplexity(pModel, summLexRank)

        print "----------KL Divergence Score ------------"
        print "KLScoreMRSvocab = "+str(KLScoreMRSvocab)
        print "KLScoreRSvocab = "+str(KLScoreRSvocab)
        print "KLScoreRSvocab = "+str(KLScoreLRvocab)
        print "------------------------------------------"

        print "----------Sentiment Analysis -------------"
        print "Senitment All Text="+str(SentimentScoreALLText)
        print "Sentiment MRS="+str(SentimentScoreMRS)
        print "Sentiment RS="+str(SentimentScoreRS)
        print "Sentiment LR="+str(SentimentScoreLR)
        print "------------------------------------------"

        print >>fidres, "------------------------------------------"
        print >>fidres, "InputFile=%s" % (inpTweetFile);
        print >>fidres, "Iter number = %d" % (iter)
        print >>fidres, "--------------Parameters------------------"
        print >>fidres, "Nsum = %d, Weights TS_w1=%f, SS_w2=%f" % (Nsumm, parameters['w1'], parameters['w2']);
        print >>fidres, "------------------------------------------"

        print >>fidres, "----------KL Divergence Score ------------"
        print >>fidres, "KLScoreMRSvocab = %s\t KLScoreRSvocab = %s\t KLScoreLRvocab = %s " % (str(KLScoreMRSvocab), str(KLScoreRSvocab), str(KLScoreLRvocab));
        print >>fidres, "------------------------------------------"

        print >>fidres, "----------Sentiment Analysis -------------"
        count=0;
        for word in keyword:
            print >>fidres, "Word = %s\tSenitment All Text = %s\tSentiment MRS= %s\tSentiment RS=%s\tSentiment LR=%s" % (word, str(SentimentScoreALLText[word][0]), str(SentimentScoreMRS[word][0]), str(SentimentScoreRS[word][0]), str(SentimentScoreLR[word][0]))
            count=count+1;
        print >>fidres, "------------------------------------------"
        print >>fidres, "-----Generated Sentiment vs Actual Sentiment"
        for word in keyword:
            print >>fidres, "Word = %s\tSenitment Generated = %s\tSentiment ACT= %s\t count=%s" % (word, str(sentimentProbGen[word]), str(sentimentProb[word]), str(sentimentHistogram[word]))
        print >>fidres, "------------------------------------------"
        print >>fidres, "----Sentiment Results---------------------"
        MRSvsACT_diff = []
        RSvsACT_diff= []
        LRvsACT_diff = []
        #MRSsentiScore=0
        #RSsentiScore=0
        ImpKeywords=parameters['sentiWordsImp'];
        for word in parameters['sentiWordsImp']:
            if (SentimentScoreALLText.has_key(word) ):
                MRSvsACT_diff.append(abs(SentimentScoreALLText[word][0]-SentimentScoreMRS[word][0]));
                RSvsACT_diff.append(abs(SentimentScoreALLText[word][0]-SentimentScoreRS[word][0]))
                LRvsACT_diff.append(abs(SentimentScoreALLText[word][0]-SentimentScoreLR[word][0]))
                if ( (abs(SentimentScoreALLText[word][0]-SentimentScoreMRS[word][0])) < (abs(SentimentScoreALLText[word][0]-SentimentScoreRS[word][0]))):
                    MRSsentiScore[iter]=MRSsentiScore[iter]+1;
                else:
                    RSsentiScore[iter]=RSsentiScore[iter]+1;
        MRSvsACT_std[iter]=l2norm(MRSvsACT_diff)/len(ImpKeywords); RSvsACT_std[iter]=l2norm(RSvsACT_diff)/len(ImpKeywords); LRvsACT_std[iter]=l2norm(LRvsACT_diff)/len(ImpKeywords)
        print >>fidres, "MRSSentiScore=%f\tRSSentiScore=%f\n" % (float(MRSsentiScore[iter]), float(RSsentiScore[iter]))
        print >>fidres, "MRS l2norm=%f\t RS l2norm=%f\t LR l2norm=%f\n" % (float(MRSvsACT_std[iter]), float(RSvsACT_std[iter]), float(LRvsACT_std[iter]))
        print >>fidres, "-----------------------------------------------------------------"

        print >>fidres, "-------------NER METRIC RESULTS------------------------------------"
        nerScoreMRS=NEReval_AllvsSumm(scorener, summTweetMRS)
        nerMRS[iter]=nerScoreMRS[1]
        nerScoreRS=NEReval_AllvsSumm(scorener, textListsummRS)
        nerRS[iter]=nerScoreRS[1]
        nerScoreLR=NEReval_AllvsSumm(scorener, summLexRank)
        nerLR[iter]=nerScoreLR[1]
        print >>fidres, "MRS NER Score=%f\tRS NER Score=%f\tLR NER Score=%f\n" % (float(nerMRS[iter]), float(nerRS[iter]), float(nerLR[iter]))

        print >>fidres, "-----------------------------------------------------------------"


        print >>fidres, "-------------VERACITY RESULTS------------------------------------"
        verscoreMRS[iter]=0;
        for indexMRS in bestTweetList:
            verscoreMRS[iter]=verscoreMRS[iter]+scorevar[indexMRS]

        verscoreRS[iter]=0;
        for indexRS in indexALLRS:
            verscoreRS[iter]=verscoreRS[iter]+scorevar[indexRS]

        verscoreLR[iter]=0;
        for indexLR in range(100):
            verscoreLR[iter]=verscoreLR[iter]+scorevar[res[indexLR]]

        print >>fidres, "verscoreMRS=%f\tverscoreRS=%f" %(verscoreMRS[iter], verscoreRS[iter])
        print >>fidres, "-----------------------------------------------------------------"

        print >>fidres, "-------------PERPLEXITY SCORE------------------------------------"
        print >>fidres, "MRS PER Score=%f\tRS PER Score=%f\tLR PER Score=%f\n" % (float(perMRS[iter]), float(perRS[iter]), float(perLR[iter]))
        print >>fidres, "-----------------------------------------------------------------"

        if (iter == (Niters-1)):
            print >>fidres, "Average MRSsentiScore = %f\tAverage RSsentiScore = %f" % (mean(MRSsentiScore), mean(RSsentiScore))
            print >>fidres, "Average MRS std-dev = %f\tAverage RS std-dev = %f\tAverage LR std-dev = %f" % (mean(MRSvsACT_std), mean(RSvsACT_std), mean(LRvsACT_std))
            print >>fidres, "Average MRS NERScore = %f\tAverage RS NERScore = %f\tAverage LR NERScore = %f" % (mean(nerMRS), mean(nerRS), mean(nerLR))
            print >>fidres, "Average MRS VeracityScore = %f\tAverage RS VeracityScore = %ftAverage LR VeracityScore = %f" % (mean(verscoreMRS), mean(verscoreRS), mean(verscoreLR))
            print >>fidres, "Average MRS perplexity = %f\tAverage RS perplexity = %ftAverage LR perplexity = %f" % (mean(perMRS), mean(perRS), mean(perLR))

        fileName='temp/ModelVlaues'+str(iter)+'.txt'
        fdpmf=open(fileName,'w');
        #for word in pModel.keys():
            #print >>fdpmf,"%f\t%f\t%f\t%f\n" % (pModel[word], pMRSModelvocab[word], sRSModelvocab[word], sLRModelvocab[word]);
        for word in pModel.keys():
            fdpmf.write("%f," % (pModel[word]))
            #print >>fdpmf,"%f," % (pModel[word]);
        fdpmf.write("\n");
        for word in pModel.keys():
            fdpmf.write("%f," % (pMRSModelvocab[word]))
            #print >>fdpmf,"%f," % (pMRSModelvocab[word]);
        fdpmf.write("\n");
        for word in pModel.keys():
            fdpmf.write("%f," % (sRSModelvocab[word]))
            #print >>fdpmf,"%f," % (sRSModelvocab[word]);
        fdpmf.write("\n");
        for word in pModel.keys():
            fdpmf.write("%f," % (sLRModelvocab[word]))
        fdpmf.write("\n");
            #print >>fdpmf,"%f," % (sLRModelvocab[word]);
        #print >>fdpmf,"%s" % (pMRSModelvocab.values());
        #print >>fdpmf,"%s" % (sRSModelvocab.values());
        #print >>fdpmf,"%s" % (sLRModelvocab.values());
        fdpmf.close()

        #Dump the summaries
        outFileName='temp/textSummMRS'+str(iter)+'.txt'
        dump_textList(summTweetMRS, outFileName)
        outFileName='temp/textSummRS'+str(iter)+'.txt'
        dump_textList(textListsummRS, outFileName)
        outFileName='temp/textSummLR'+str(iter)+'.txt'
        dump_textList(summLexRank, outFileName)



    fidres.close()
