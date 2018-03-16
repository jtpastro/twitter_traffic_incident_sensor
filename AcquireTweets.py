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

def loadFreq(freqFilename, tweets, ngram=2, limit=100):
    jsonPath = Path(freqFilename)
    if jsonPath.exists():
        with jsonPath.open() as jsonFile:
            j = json.load(jsonFile)
        return j.items()
    else:
        with jsonPath.open('w') as jsonFile:
            tokens = tokenizeTweets(tweets) 
            freq = dict((' '.join(w),f) for n in range(ngram, 0, -1) for w,f in FreqDist(ngrams(tokens, n)).most_common(limit))
            json.dump(freq, jsonFile)
        return freq

def tokenizeTweets(tweets):
    tknzr = RegexpTokenizer(r'[\w-]+')
    stopwordList = stopwords.words('portuguese')
    return [word for word in tknzr.tokenize(" ".join(tweets).lower()) if word not in stopwordList and len(word) > 3 and not word[0].isdigit()]

def bestGrams(allFreq):
    freq = {}
    for w,f in allFreq:
        if not any(set(w).issubset(o) for o in freq):
            freq[w] = f
    return freq

if __name__ == '__main__':
    tFile = 'merged.json'
    fFile = 'freq.json'
    query = 'from:marinapagno OR from:EPTC_POA'
    tweets = loadTweets(tFile, query)
    freq = bestGrams(loadFreq(fFile, tweets))
    for w in freq:
        print(w+' => '+str(freq[w]))

    
