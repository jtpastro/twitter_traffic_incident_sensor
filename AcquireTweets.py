from twitterscraper import query_tweets
from nltk import FreqDist
from nltk.util import ngrams
import TokenizeTweet as tt
from FileLoader import openElseLoad

def loadTweets(query, maxResults=None):
    return [tweet.text for tweet in query_tweets(query, limit=maxResults)]

def loadFreq(tokens, ngram=2, limit=400):
    return bestGrams((' '.join(w),f) for n in range(ngram, 0, -1) for w,f in FreqDist(ngrams(tokens, n)).most_common(limit))

def bestGrams(allFreq):
    freq = {}
    for w,f in allFreq:
        if not any(set(w).issubset(o) for o in freq):
            freq[w] = f
    return freq

if __name__ == '__main__':
    
    tFile = 'cache/tweets_traffic.json'
    fFile = 'cache/freq_words_traffic.json'
    query = 'from:marinapagno OR from:EPTC_POA'
    lt = lambda: loadTweets(query)
    tweets = openElseLoad(tFile, lt)
    tokens = [token for tweet in tweets for token in tt.tokenizeTweet(tweet)] 
    lf = lambda: loadFreq(tweets)
    freq = openElseLoad(fFile, lf)
    for w in freq:
        lt = lambda: loadTweets(w+' near:"Porto Alegre"', 800)
        tweets = openElseLoad('tweets/'+w.replace(' ', '_')+'.json', lt)