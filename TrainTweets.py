from pathlib import Path
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from AcquireTweets import tokenizeTweet

def loadTrainingSet(initialPath):
    for path in Path(initialPath).iterdir():
        for path3 in zip(*[(path3 for path3 in Path(path2).iterdir()) for path2 in Path(path).iterdir()]):
            yield path3

def loadTaggedDocuments(classifiedTweets):
    for tweetClass in classifiedTweets:
        for tweetFile in tweetClass:
            text = ""
            with tweetFile.open(encoding="utf8") as f:
                text += f.readline()
            yield TaggedDocument(tokenizeTweet(text), str(tweetFile).split('\\')[-2])

loadTaggedDocuments(zip(*[i for i in loadTrainingSet('tweets/Classificados/')]))