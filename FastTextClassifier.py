from WordEmbeddingClassifier import WordEmbeddingClassifier
from pathlib import Path
import fasttext as ft

class FastTextClassifier(WordEmbeddingClassifier):
    def __init__(self, trainingFile, vecSize=140, winSize=5, epochs=125, minCount=0, lossFunction=-5, sampleThreshold=0.001, learnRate=0.025, modelFile=None):
        self.trainingFile = trainingFile
        self.modelFile = modelFile if modelFile else str(Path(trainingFile).with_suffix(""))
        self.train(vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate)
    def train(self, vecSize, winSize, epochs, minCount, lossFunction, sampleThreshold, learnRate):
        self.model = ft.supervised(input_file=self.trainingFile,
                                    output=self.modelFile,
                                    lr=learnRate,
                                    dim=vecSize,
                                    ws=winSize,
                                    epoch=epochs,
                                    min_count=minCount,
                                    loss=("ns" if lossFunction < 0 else ("softmax" if lossFunction == 0 else "hs")),
                                    neg=(-lossFunction if lossFunction < 0 else 0),
                                    t=sampleThreshold,
                                    thread=8
                        )
    def classify(self, sentence):
        return self.model.predict_proba(sentence, 1)
    def evaluate(self, testFile):
        result = self.model.test(testFile, 1)
        return result.precision, result.recall

if __name__ == '__main__':
    labelled = "cache/ft_labelled_tweets.txt"
    ft = FastTextClassifier(labelled)
    _eval = ft.evaluate(labelled)
    print(_eval)
    from TokenizeTweet import tokenizeTweet
    tweets = ["o transito na ipiranga esta lento", "o trafego na internet esta lento", "o trafego de rede esta lento"]
    t = [" ".join([t for t in tokenizeTweet(twt)]) for twt in tweets]
    print(t, ft.classify(t))