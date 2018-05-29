import pickle
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from TokenizeTweet import getStopwords, tokenizeTweet
import collections
from numpy import dot
from numpy.linalg import norm
import fasttext as ft

def loadTrainingSet(initialPath):
    for path in Path(initialPath).iterdir():
        for path3 in zip(*[(path3 for path3 in Path(path2).iterdir()) for path2 in Path(path).iterdir()]):
            yield path3

def loadTaggedDocuments(classifiedTweets):
    i=0
    for tweetClass in classifiedTweets:
        for tweetFile in tweetClass:
            text = ""
            with tweetFile.open(encoding="utf8") as f:
                text = f.readline()
            i+=1
            yield TaggedDocument([token for token in tokenizeTweet(text)], [tweetFile.parent.name, "SENT_"+str(i)])

def trainDoc2Vec(trainingSet, path=None, loadFile=True):
    if loadFile and path and Path(path).exists():
        return Doc2Vec.load(path)
    model = Doc2Vec(dm=0, alpha=0.025, vector_size=140, min_alpha=0.0001, epochs=125, min_count=0, workers=8)
    model.build_vocab(trainingSet)
    model.train(trainingSet, total_examples=model.corpus_count, epochs=model.epochs)
    if path:
        model.save(path)
    return model

def ranking(train_corpus, model):
    ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(train_corpus[doc_id].tags[1])
        ranks.append(rank)

    mode, freq = zip(*collections.Counter(ranks).most_common(3))
    mean = sum(ranks)/len(ranks)
    median = sorted(ranks)[len(ranks)//2]
    return mode, int(mean+0.5), median, int(sum(freq)*100/len(ranks)+0.5) 

def selfAssessment(train_corpus, model):
    for doc in train_corpus:
        inferred_vector = model.infer_vector(doc.words)
        yield doc.tags[0], cosineDistance(model["transit"], inferred_vector), cosineDistance(model["not_transit"], inferred_vector)

def cosineDistance(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def testDoc2Vec(trainingSet, tweets):
    model = trainDoc2Vec(trainingSet, "cache/trainedModel.bin", False)
    print("-------")
    print("DOC2VEC")
    print("-------")
    #print("Mode: {stat[0]}, Mean: {stat[1]}, Median: {stat[2]}, Match: {stat[3]}%".format(stat=ranking(trainingSet, model)))
    res = collections.defaultdict(int)
    for tag, t_cls, nt_cls in selfAssessment(trainingSet, model):
        if (tag=="transit") == (t_cls > nt_cls):
            res[tag] += 1
    for tag in res:
        print(tag, "{:.2f}%".format(res[tag]*200/len(trainingSet)))

    for twt in tweets:
        tkns = [w for w in tokenizeTweet(twt)]
        iv = model.infer_vector(tkns)
        print(tkns)
        print(cosineDistance(model["transit"], iv))
        print(cosineDistance(model["not_transit"], iv))
    print("-------")

def testFastText(trainFile, modelFile, tweets):
    print("-------")
    print("FASTTEXT")
    print("-------")
    classifier = ft.load_model(modelFile) if Path(modelFile).exists() else ft.supervised(trainFile, modelFile)
    for twt in tweets:
        print(classifier.predict([" ".join([w for w in tokenizeTweet(twt)])]))
        print("-------")
    result = classifier.test(trainFile)
    print ('P@1:', result.precision)
    print ('R@1:', result.recall)
    print ('Number of examples:', result.nexamples)

def loadFastTextFormat(ftFile):
    tdList = []
    if ftFile.exists():
        with ftFile.open(mode='r') as f:
            for line in f:
                tokens = line.split()
                tags = []
                words = []
                for token in tokens:
                    if "__label__" in token:
                        tags.append(token.split("__label__")[-1])
                    else:
                        words.append(token)
                    tdList.append(TaggedDocument(words, tags))
    else:
        with ftFile.open(mode='w') as f:
            sId = 0
            for tweetClass in zip(*[i for i in loadTrainingSet('tweets/Classificados/')]):
                for tweetFile in tweetClass:
                    with tweetFile.open(encoding="utf8") as f2:
                        text = f2.readline()
                    sId += 1
                    words = [tkn for tkn in tokenizeTweet(text)]
                    tokens = [" ".join(words), "__label__" + tweetFile.parent.name, "__label__"+str(sId)]
                    tdList.append(TaggedDocument(words, tokens[1:]))
                    f.write(" ".join(tokens)+'\n')
    return tdList

if __name__ == '__main__':
    ftFilename = 'cache/ft_labelled_tweets.txt'
    #ftModelFile = 'cache/trainedModelFT'
    trainingSet = loadFastTextFormat(Path(ftFilename))
    #trainingSet = pickle.load(open('cache/trainingSet.pickle', 'rb'))
    #tweets = ["o transito na ipiranga esta lento", "o trafego na internet esta lento", "o trafego de rede esta lento"]
    #testDoc2Vec(trainingSet, tweets)
    #testFastText(ftFilename, ftModelFile, tweets)
