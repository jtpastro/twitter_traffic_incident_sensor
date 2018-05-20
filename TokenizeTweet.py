from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, floresta
from nltk import FreqDist
from unidecode import unidecode as unicodeToAscii
from FileLoader import openElseLoad

def tokenizeTweet(tweet, stopwordList=openElseLoad('cache/stopwords.json', lambda: getStopwords())):
    tknzr = RegexpTokenizer(r'[\w-]+')
    for word in tknzr.tokenize(unicodeToAscii(tweet.lower())):
        if word not in stopwordList and len(word)>1 and not word[0].isdigit():
            yield word

def getStopwords():
    return stopwords.words('portuguese') + [w.lower() for w,f in FreqDist(floresta.words()).most_common(500) if len(w)<=3]