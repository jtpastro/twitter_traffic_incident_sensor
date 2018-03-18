from twitterscraper import query_tweets
from pathlib  import Path
import json
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, floresta
from nltk.util import ngrams

def openElseLoad(filename, loadFunction):
    jsonPath = Path(filename)
    if jsonPath.exists():
        with jsonPath.open() as jsonFile:
            return json.load(jsonFile)
    else:
        with jsonPath.open('w') as jsonFile:
            data = loadFunction()
            json.dump(data, jsonFile)
        return data


def loadTweets(tweetFilename, query, maxResults=None):
    tweets = lambda: [tweet.text for tweet in query_tweets(query, limit=maxResults)]
    return openElseLoad(tweetFilename, tweets)

def loadFreq(freqFilename, tweets, ngram=2, limit=400):
    tokens = [token for tweet in tweets for token in tokenizeTweet(tweet)] 
    freq = lambda: bestGrams((' '.join(w),f) for n in range(ngram, 0, -1) for w,f in FreqDist(ngrams(tokens, n)).most_common(limit))
    return openElseLoad(freqFilename, freq)

def tokenizeTweet(tweet):
    tknzr = RegexpTokenizer(r'[\w-]+')
    stopwordList = getStopwords()
    return [word for word in tknzr.tokenize(tweet.lower()) if word not in stopwordList and len(word)>1 and not word[0].isdigit()]

def bestGrams(allFreq):
    freq = {}
    for w,f in allFreq:
        if not any(set(w).issubset(o) for o in freq):
            freq[w] = f
    return freq

def getStopwords():
    swFunc = lambda: stopwords.words('portuguese') + [w.lower() for w,f in FreqDist(floresta.words()).most_common(500) if len(w)<=4]
    return openElseLoad('stopwords.json', swFunc)

def fileLinesToList(filename):
    try:
        with open(filename) as _file:
            return _file.readlines()
    except:
        return []

if __name__ == '__main__':
    tFile = 'cache/tweets_traffic.json'
    fFile = 'cache/freq_words_traffic.json'
    query = 'from:marinapagno OR from:EPTC_POA'
    tweets = loadTweets(tFile, query)
    freq = loadFreq(fFile, tweets)
    for w in freq:
        loadTweets('tweets/'+w.replace(' ', '_')+'.json', w+' near:"Porto Alegre"', 800)


    
