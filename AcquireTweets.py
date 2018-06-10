from twitterscraper import query_tweets
from nltk import FreqDist
from nltk.util import ngrams
from TokenizeTweet import FilterTokenizer
from FileLoader import openElseLoad
from pathlib import Path

def loadFreq(tweets, ngram=2, limit=400):
    return bestGrams((' '.join(w),f) for tweet in tweets for n in range(ngram, 0, -1) for w,f in FreqDist(ngrams(tt.tokenize(tweet.text), n)).most_common(limit))

def bestGrams(allFreq):
    freq = {}
    for w,f in allFreq:
        if not any(set(w).issubset(o) for o in freq):
            freq[w] = f
    return freq

def saveTweets(keyword, path, tweets):
    path = Path(path) / keyword
    transit = path / "transit"
    transit.mkdir(parents=True, exist_ok=True)
    not_transit = path / "not_transit"
    not_transit.mkdir(parents=True, exist_ok=True)
    for i, twt in enumerate(tweets):
        tweetPath = path / (str(i)+'.txt')
        with tweetPath.open('w', encoding="utf-8") as tweetFile:
            tweetFile.write(twt.text)

if __name__ == '__main__':
    fFile = 'cache/freq_words_traffic2.json'
    query = 'from:marinapagno OR from:EPTC_POA'
    tt = FilterTokenizer() 
    tweets = query_tweets(query)
    lf = lambda: loadFreq(tweets)
    freq = openElseLoad(fFile, lf)
    for w in freq:
        lt = query_tweets(w+' near:"Porto Alegre"', 800)
        saveTweets(w, 'tweets/Classificados 2/', lt)