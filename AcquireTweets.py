from twitterscraper import query_tweets
from pathlib  import Path
import json
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
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

def loadFreq(freqFilename, tweets, ngram=1, limit=1000):
    jsonPath = Path(freqFilename)
    if jsonPath.exists():
        with jsonPath.open() as jsonFile:
            return json.load(jsonFile)
    else:
        with jsonPath.open('w') as jsonFile:
            tknzr = RegexpTokenizer(r'[\w-]+')
            stopwordList = stopwords.words('portuguese')
            freq = FreqDist(ngrams([word for word in tknzr.tokenize(" ".join(tweets).lower()) if word not in stopwordList and len(word) > 3 and not word[0].isdigit()], ngram)).most_common(limit)
            json.dump(freq, jsonFile)
        return freq

if __name__ == '__main__':
    tFile = 'merged.json'
    fFile = 'freq.json'
    query = 'from:marinapagno OR from:EPTC_POA'
    tweets = loadTweets(tFile, query)
    freq = loadFreq(fFile, tweets, 3)
    for w,f in freq:
        print(' '.join(w)+' => '+str(f))
