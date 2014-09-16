import pdb;
import json


def getSentimentAll(textList):
    posScore = 0;
    negScore = 0;
    neutScore= 0;
    for i in range(len(textList)):
        words = textList[i].split(' ')
        score[i]=0
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
    return sentiValue

def getSentimentKeywords(textList,keyword):
    score = 0
    positive={}
    negative={}
    neutral={}
    count={}
    textSenti = ['positive','negative','neutral']

    sentiValue = [[0 for i in range(len(textSenti))] for j in range(len(keyword))]

    for k in range(len(keyword)):
        count[keyword[k]]=0
        positive[keyword[k]]=0
        negative[keyword[k]]=0
        neutral[keyword[k]]=0

    for i in range(len(textList)):
        line =  textList[i]
        for j in range(len(keyword)):
            if keyword[j] in line:
                count[keyword[j]]+=1
                words = line.split(' ')
                for word in words:
                    try:
                        score = sentiDict[word]
                        if score>0:
                            positive[keyword[j]]+=1
                        elif score<0:
                            negative[keyword[j]]+=1
                    except KeyError:
                        score =0
                        neutral[keyword[j]]+=1


    for i in range(len(keyword)):
        neutral[keyword[i]] = count[keyword[i]]-positive[keyword[i]]-negative[keyword[i]]

    for i in range(len(keyword)):
            sentiValue[i][0] = positive[keyword[i]]/float(count[keyword[i]])*100
            sentiValue[i][1] = negative[keyword[i]]/float(count[keyword[i]])*100
            sentiValue[i][2] = neutral[keyword[i]]/float(count[keyword[i]])*100
    return sentiValue



if __name__=="__main__":
    clean=lambda orig: "".join([x for x in orig if ord(x) < 128])
    values = []
    f_json = open('tweets.txt','r+')
    for line in f_json:
        values.append(json.loads(line))


    f = open('senti_words.txt','r')
    sentiDict={}
    for line in f:
        new = line.split('\t')
        #pdb.set_trace();
        if (len(new)>1):
            sentiDict[new[0].rstrip()]=int(new[1].rstrip())
        else:
            sentiDict[new[0].rstrip()]=0
    textList = []
    for i in range(len(values)):
        textList.append(values[i]['text'])
    score = {}
    sentiValue = {}
    sentiValue = getSentimentAll(textList)
    print sentiValue

    keyword = ['Modi','Rahul','Kejriwal']
    print getSentimentKeywords(textList,keyword)
