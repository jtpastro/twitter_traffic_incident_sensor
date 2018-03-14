from twitterscraper import query_tweets
from pathlib  import Path
import json
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 
from nltk.util import ngrams

def loadTweets(tweetFilename, query):
    jsonPath = Path(tweetFilename)
    if jsonPath.exists():
        with jsonPath.open() as jsonFile:
            return json.load(jsonFile)
    else:
        with jsonPath.open('w') as jsonFile:
            tweets = [tweet.text for tweet in query_tweets(query)]
            json.dump(tweets, jsonFile)
        return tweets

def loadFreq(freqFilename, tweets, ngram=1, limit=100):
    jsonPath = Path(freqFilename)
    if jsonPath.exists():
        with jsonPath.open() as jsonFile:
            return json.load(jsonFile)
    else:
        with jsonPath.open('w') as jsonFile:
            tknzr = RegexpTokenizer(r'[\w-]+')
            stopwordList = stopwords.words('portuguese')
            freq = FreqDist(ngrams((word for word in tknzr.tokenize(" ".join(tweets).lower()) if word not in stopwordList and len(word) > 3 and not word[0].isdigit()), ngram)).most_common(limit)
            json.dump(freq, jsonFile)
        return freq

if __name__ == '__main__':
    tFile = 'merged.json'
    fFile = 'freq.json'
    query = 'from:marinapagno OR from:EPTC_POA'
    query2 = 'from:marinapagno OR from:EPTC_POA'
    tweets = loadTweets(tFile, query)
    freq = {}
    for i in range(2,0,-1):
        for w,f in loadFreq(fFile[:4]+str(i)+fFile[4:], tweets, i):
            hasBiggerNGram = False
            w = ' '.join(w)
            for o in freq:
                if(set(w).issubset(o)):
                    hasBiggerNGram = True
                    break
            if not hasBiggerNGram:
                freq[w] = f

    for w in freq:
        print(w+' => '+str(freq[w]))
