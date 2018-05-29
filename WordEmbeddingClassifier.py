

class WordEmbeddingClassifier:
    def __init__(self, trainingFile, vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate):
        raise NotImplementedError
    def train(self, vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate):
        raise NotImplementedError
    def classify(self, sentence):
        raise NotImplementedError
    def evaluate(self, testFile):
        raise NotImplementedError
    
