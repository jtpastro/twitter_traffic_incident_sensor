from pathlib import Path
from collections import defaultdict
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import dot
from numpy.linalg import norm
from WordEmbeddingClassifier import WordEmbeddingClassifier

def cosineDistance(a,b):
    return dot(a, b)/(norm(a)*norm(b))

class Paragraph2VecClassifier(WordEmbeddingClassifier):
    def __init__(self, trainingFile, vecSize=140, winSize=5, epochs=125, minCount=0, lossFunction=-5, sampleThreshold=0.001, learnRate=0.025):
        self.trainingData = [td for td in self.loadTrainingData(trainingFile)]
        self.train(vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate)

    def loadTrainingData(self, trainingFile):
        self.tags = set()
        with Path(trainingFile).open(mode='r') as f:
            for line in f:
                tokens = line.split()
                localTags = []
                words = []
                for token in tokens:
                    if "__label__" in token:
                        tag = token.split("__label__")[-1]
                        localTags.append(tag)
                        if not tag.isnumeric():
                            self.tags.add(tag)
                    else:
                        words.append(token)
                yield TaggedDocument(words, localTags)

    def train(self, vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate):
        self.model = Doc2Vec(documents=self.trainingData,
                                dm=0,
                                vector_size=vecSize,
                                alpha=learnRate,
                                min_alpha=1e-4,
                                min_count=minCount,
                                epochs=epochs,
                                window=winSize,
                                hs= (1 if lossFunction > 0 else 0),
                                negative=(-lossFunction if lossFunction < 0 else 0),
                                sample=sampleThreshold)

    def classify(self, sentence):
        iv = self.model.infer_vector(sentence)
        return {tag: cosineDistance(self.model[tag], iv) for tag in self.tags}

    def evaluate(self, testFile):
        correctCounter = defaultdict(int) #tp
        classifierCounter = defaultdict(int) #tp+fp
        taggedCounter = defaultdict(int) #tp+fn
        for td in self.loadTrainingData(testFile):
            tagSimilarityDict = {}
            for tag in self.tags:
                if tag in td.tags:
                    taggedCounter[tag] += 1
                inferred_vector = self.model.infer_vector(td.words)
                tagSimilarityDict[tag] = cosineDistance(self.model[tag], inferred_vector)
            bestTag = max(tagSimilarityDict, key=tagSimilarityDict.get)
            classifierCounter[bestTag] += 1
            if bestTag in td.tags:
                correctCounter[bestTag] += 1
        return {tag: {"precision": correctCounter[tag]/classifierCounter[tag], "recall": correctCounter[tag]/taggedCounter[tag]} for tag in tags}

if __name__ == '__main__':
    labelled = "cache/ft_labelled_tweets.txt"
    p2vc = Paragraph2VecClassifier(labelled)
    _eval = p2vc.evaluate(labelled)
    print(_eval)
    from TokenizeTweet import tokenizeTweet
    tweets = ["o transito na ipiranga esta lento", "o trafego na internet esta lento", "o trafego de rede esta lento"]
    for twt in tweets:
        t = [t for t in tokenizeTweet(twt)]
        print(t, p2vc.classify(t))