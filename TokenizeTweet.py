from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, floresta
from nltk import FreqDist
from unidecode import unidecode as unicodeToAscii
from FileLoader import openElseLoad

def tokenizeTweet(tweet, stopwordList=None):
    if not stopwordList:
        stopwordList = openElseLoad('cache/stopwords.json', lambda: getStopwords())
    tknzr = RegexpTokenizer(r'[\w-]+')
    for word in tknzr.tokenize(tweet.lower()):
        if word not in stopwordList and len(word)>1 and not word[0].isdigit():
            yield word

def getStopwords():
    return stopwords.words('portuguese') + [w.lower() for w,f in FreqDist(floresta.words()).most_common(500) if len(w)<=3]

def tokenizeTweet2(tweet, stopwordList=None):
    if not stopwordList:
        stopwordList = openElseLoad('cache/stopwords.json', lambda: getStopwords2())
    tknzr = RegexpTokenizer(r'[a-zA-Z-]+')
    for word in tknzr.tokenize(unicodeToAscii(tweet.lower())):
        if word not in stopwordList and len(word)>1:
            yield word

def getStopwords2():
    return [unicodeToAscii(w.lower()) for w in stopwords.words('portuguese')] + [unicodeToAscii(w.lower()) for w,f in FreqDist(floresta.words()).most_common(500) if len(w)<=3] + ["60", "mais", "pic", "twitter", "http", "htm", "php", "www", "rt", "default", "gov", "www2", "https", "rdgauchapic", "gauchatransitopic", "gauchazhpic", "p_noticia", "ta", "tah", "swarm", "apos", "z3pyhh77eh", "bvw_k8dfkgu"]

if __name__ == '__main__':
    print(openElseLoad('cache/stopwords.json', lambda: getStopwords2()))