#from numpy import *
import math
import numpy
import nltk
import sys
import json
from pprint import pprint
from collections import Counter
import pdb
import UsefulFuncs

clean=lambda orig: "".join([x for x in orig if ord(x) < 128])

class BagOfWords:
    def __init__(self,stem=True):
        self.vocabulary={}
        self.num_documents=0
        self.stoplist={}
        self.stem2orig={}
        self.prob={}

        if stem:
            self.stemmer=nltk.stem.PorterStemmer() #lovins
        else:
            self.stemmer=nltk.stem.PorterStemmer()
            self.stemmer.stem = lambda x: x

        self.stemmer.unstem = lambda x: self.stemdict.has_key(x) and self.stemdict[x][0][1] or x


        #stoplist=nltk.corpus.stopwords.words()
        #for word in stoplist:
        #    self.stoplist[self.stemmer.stem(word)]=1

        self.index={}
        self.idf={}
        self.tokenizer=nltk.tokenize.RegexpTokenizer('[a-z]\w+')
        self.weights={}
        self.bias=0

    def getTFMatrix(self, textList):

        for i in range(len(textList)):
            words=self.tokenizer.tokenize(textList[i].lower())
            textWords=[self.stemmer.stem(clean(word1)) for word1 in words]

            TFMatrix=[[0 for row in range(len(textList))] for col in range(len(self.vocabulary))]
            j=0
            for wordVocab in self.vocabulary:
                print"i="+str(i)+"j="+str(j)
                TFMatrix[j][i]= textWords.count(wordVocab)
                j=j+1

        return TFMatrix;

    def addstoplist(self,stoplist):
        for word in stoplist:
            self.stoplist[self.stemmer.stem(word)]=1

    def build_index(self,text_iterator,prune_min):
        for text in text_iterator:
            self.num_documents+=1
            words={}
            for word in self.tokenizer.tokenize(text.lower()):
                cleaned_word=clean(word)
                stemmed_word=self.stemmer.stem(cleaned_word)
                words[stemmed_word]=1
                if self.stem2orig.has_key(stemmed_word):
                    if self.stem2orig[stemmed_word].has_key(cleaned_word):
                        self.stem2orig[stemmed_word][cleaned_word] =  self.stem2orig[stemmed_word][cleaned_word]+1
                    else:
                        self.stem2orig[stemmed_word][cleaned_word] = 1
                else:
                    self.stem2orig[stemmed_word]={}
                    self.stem2orig[stemmed_word][cleaned_word]=1

            #words=dict.fromkeys([self.stemmer.stem(clean(word)) for word in self.tokenizer.tokenize(text.lower())],1)
            #pdb.set_trace();
            for word in words.keys():
                if not(self.stoplist.has_key(word)):
                    if self.vocabulary.has_key(word):
                        self.vocabulary[word]+=1
                    else:
                        self.vocabulary[word]=1
        prune_words=[word for word in self.vocabulary if self.vocabulary[word]<prune_min]
        for word in prune_words:
            del self.vocabulary[word]
        t=1
        vocab=self.vocabulary.keys()
        vocab.sort()
        for word in vocab:
            self.index[word]=t
            t+=1

        for word in self.vocabulary.keys():
            self.idf[word]=math.log(self.num_documents*1.0/self.vocabulary[word])

    def est_probability_UD(textList):
        vocabFile='vocabUD.txt';
        vocabUD=[];
        with open(vocabFile,'r') as vFile:
            for line in vFile:
                vocabUD.append(line);
        vocabUD_dict={};
        for i in range(len(textList)):
            textWords=textList[i].split(" ");
            for word in vocabUD:
                count=textWords.count(word);
                if word in vocabUD_dict:
                    vocabUD_dict[word]=vocabUD_dict[word]+count;
                else:
                    vocabUD_dict[word]=count;
        sumDict=0
        for word in vocabUD_dict:
            sumDict=sumDict+vocabUD_dict[word]
        pDict={}
        for word in vocabUD_dict.keys():
            pDict[word.lower().rstrip()]=float(vocabUD_dict[word])/sumDict
            #print "word= "+word+" Prob="+str(self.prob[word.lower().rstrip()])
        #if (UsefulFuncs.trunc((sum(self.prob.values())),4) != 1.0):
        if (not(UsefulFuncs.feq(sum(pDict.values()),1.0))):
            raise Exception('Probability Error: pdf doesnt sum to one');

        return pDict;

    def set_index(self,index,idf):
        self.index=index;
        self.idf=idf;

    def add_stem2orig(self, stem2orig):
         self.stem2orig = dict(self.stem2orig, **stem2orig)

    def set_weights(self,weights,bias):
        self.weights=weights
        self.bias=bias

    def term_freq(self,text):
        tf={}
        for word in self.tokenizer.tokenize(text.lower()):
            word=self.stemmer.stem(clean(word))
            if self.index.has_key(word):
                if tf.has_key(word):
                    tf[word]+=1
                else:
                    tf[word]=1
        return tf

    def word_features(self,text,type='tf',normalize=True):
        tf=self.term_freq(text)

        if type=='tf':
            features=tf
        if type=='tfidf':
            features={}
            for word in tf.keys():
                features[word]=tf[word]*self.idf[word]
        if type=='ltc':
            features={}
            for word in tf.keys():
                features[word]=(1+math.log(tf[word]))*self.idf[word]
        if normalize:
            norm=math.sqrt(sum([features[a]*features[a] for a in features]))
            for a in features:
                features[a]=features[a]*1.0/norm
        return features

    def dump_sparse_vector(self,text,type='tf',normalize=True,fid=sys.stdout):
        features=self.word_features(text,type,normalize)
        tuples=[(self.index[word],features[word]) for word in features]
        tuples.sort()
        for (idx,val) in tuples:
            print>>fid, "%d:%f" % (idx,val),
        print>>fid

    def dump_vocab(self,fid):
        tuples=[(self.index[word],word) for word in self.index]
        tuples.sort()
        print len(tuples)
        for idx,val in tuples:
            print >>fid,"%d\t%s\t%f" % (idx,val,self.idf[val])

    def dump_tf(self, fid):
	   for word in self.vocabulary:
            print >>fid,"%s\t%f" % (word,self.vocabulary[word])

    def est_probability(self,Nterms):
        prunDict=dict(Counter(self.vocabulary).most_common()[:Nterms])
        #prunDict={}

        sumDict=0
        for word in prunDict:
            sumDict=sumDict+prunDict[word]
        for word in prunDict:
            self.prob[word.lower().rstrip()]=float(prunDict[word])/sumDict
            #print "word= "+word+" Prob="+str(self.prob[word.lower().rstrip()])
        #if (UsefulFuncs.trunc((sum(self.prob.values())),4) != 1.0):
        if (not(UsefulFuncs.feq(sum(self.prob.values()),1.0))):
            raise Exception('Probability Error: pdf doesnt sum to one')
        return self.prob

    def vocabsize(self):
        return len(self.index)

    def dump_unstemmed_vocab(self,fid):
        tuples=[(self.index[word],word) for word in self.index]
        tuples.sort()
        for idx,val in tuples:
            idf=self.idf[val]
            if self.stem2orig.has_key(val):
                candidates=[(v,k) for (k,v) in self.stem2orig[val].items()]
                candidates.sort()
                candidates.reverse()
                val=candidates[0][1]
            print >>fid,"%d\t%s\t%f" % (idx,val,idf)

    def dump_idf(self):
        print self.idf

    def dump_df(self):
        print self.vocabulary


    def classify(self,text,type='tf',normalize=True):
        features=self.word_features(text,type,normalize)
        score=0.0
        for word in features:
            score+=self.weights[word]*features[word]
        score+=self.bias
        return score

    def explain_score(self,text,type='tf',normalize=True):
        features=self.word_features(text,type,normalize)
        score=0.0
        diagnostics=[]
        for word in features:
            x1=self.weights[word]
            x2=features[word]
            x=x1*x2
            score+=x
            diagnostics.append((x,word,x2,x1,))

        diagnostics.sort()
        diagnostics.reverse()
        for (x,word,x2,x1) in diagnostics:
            print "word=%s \t contribution=%f \t feature_value=%f \t model_weight=%f" % (word, x, x2,x1)
        score+=self.bias
        print "score=%f, bias=%f" % (score,self.bias)
        return score
