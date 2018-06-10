from pathlib import Path
import fastText.FastText as ft
from pathlib import Path
from collections import defaultdict
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import dot
from numpy.linalg import norm

class WordEmbeddingClassifier:
    def __init__(self, trainingFile, vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate):
        raise NotImplementedError
    def train(self, vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate):
        raise NotImplementedError
    def classify(self, sentence):
        raise NotImplementedError
    def evaluate(self, testFile):
        raise NotImplementedError

class FastTextClassifier(WordEmbeddingClassifier):
    def __init__(self, trainingFile, vecSize=50, winSize=5, epochs=20, minCount=5, lossFunction=-5, sampleThreshold=0.001, learnRate=0.025, modelFile=None, ngrams=0, wordGrams=1, bucket=2000000):
        self.trainingFile = trainingFile
        self.modelFile = modelFile
        self.train(vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate, ngrams, wordGrams, bucket)

    def train(self, vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate, ngrams, wordGrams, bucket):
        self.model = ft.train_supervised(input=self.trainingFile,
                                    lr=learnRate,
                                    dim=vecSize,
                                    ws=winSize,
                                    epoch=epochs,
                                    minCount=minCount,
                                    loss=("ns" if lossFunction < 0 else ("softmax" if lossFunction == 0 else "hs")),
                                    neg=(-lossFunction if lossFunction < 0 else 0),
                                    t=sampleThreshold,
                                    minn=ngrams//2,
                                    maxn=ngrams,
                                    wordNgrams=wordGrams,
                                    bucket=bucket,
                                    verbose=0,
                                    thread=4#8
                        )
        
    def classify(self, sentence):
        return self.model.predict(sentence)

    def evaluate(self, testFile):
        result = self.model.test(testFile)
        return {"combined": {"precision": result[1], "recall": result[2]}}

def cosineDistance(a,b):
    return dot(a, b)/(norm(a)*norm(b))

class Paragraph2VecClassifier(WordEmbeddingClassifier):
    def __init__(self, trainingFile, vecSize=50, winSize=5, epochs=20, minCount=5, lossFunction=-5, sampleThreshold=0.01, learnRate=0.0025, labelMarker="__label__", trainingAlgorithm=0, learnDropRate=1/25):
        self.labelMarker = labelMarker
        self.trainingData = [td for td in self.loadTrainingData(trainingFile)]
        self.train(vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate, trainingAlgorithm, learnDropRate)

    def loadTrainingData(self, trainingFile):
        self.tags = set()
        with Path(trainingFile).open(mode='r') as f:
            for line in f:
                tokens = line.split()
                localTags = []
                words = []
                for token in tokens:
                    if self.labelMarker in token:
                        tag = token.split(self.labelMarker)[-1]
                        localTags.append(tag)
                        if not tag.isnumeric():
                            self.tags.add(tag)
                    else:
                        words.append(token)
                yield TaggedDocument(words, localTags)

    def train(self, vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate, trainingAlgorithm, learnDropRate):
        """
        trainingAlgorithm 0: dbow, 1: dmcat, 2: dm_sum, 3: dm_mean
        """
        self.model = Doc2Vec(documents=self.trainingData,
                                dm=1 if trainingAlgorithm > 0 else 0,
                                vector_size=vecSize,
                                alpha=learnRate,
                                min_alpha=learnRate*learnDropRate,
                                min_count=minCount,
                                epochs=epochs,
                                window=winSize,
                                hs= (1 if lossFunction > 0 else 0),
                                negative=(-lossFunction if lossFunction < 0 else 0),
                                sample=sampleThreshold,
                                dm_concat=1 if trainingAlgorithm == 1 else 0,
                                dm_mean=trainingAlgorithm%2,
                                workers=4#8
                                )

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
        #print(correctCounter, classifierCounter, taggedCounter)
        return {tag: {"precision": correctCounter[tag]/max(classifierCounter[tag],1), "recall": correctCounter[tag]/taggedCounter[tag]} for tag in self.tags}

if __name__ == '__main__':
    labelled = "cache/train.txt"
    ft = FastTextClassifier(labelled)
    _eval = ft.evaluate(labelled)
    print(_eval)
    from TokenizeTweet import FilterTokenizer
    tknzr = FilterTokenizer()
    tweets = ["o transito na ipiranga esta lento", "o trafego na internet esta lento", "o trafego de rede esta lento"]
    t = [" ".join([t for t in tknzr.tokenize(twt)]) for twt in tweets]
    print(t, ft.classify(t))
    labelled = "cache/label_tweets_train.txt"
    p2vc = Paragraph2VecClassifier(labelled)
    _eval = p2vc.evaluate(labelled)
    print(_eval)
    tweets = ["o transito na ipiranga esta lento", "o trafego na internet esta lento", "o trafego de rede esta lento"]
    for twt in tweets:
        t = [t for t in tknzr.tokenize(twt)]
        print(t, p2vc.classify(t))