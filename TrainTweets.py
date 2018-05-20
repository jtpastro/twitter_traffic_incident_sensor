import pickle
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from TokenizeTweet import getStopwords, tokenizeTweet
import collections
from numpy import dot
from numpy.linalg import norm

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

def train(trainingSet, path=None, loadFile=True):
    #doc2vec parameters
    if loadFile and path and Path(path).exists():
        return Doc2Vec.load(path)
    model = Doc2Vec(dm=0, alpha=0.025, vector_size=140, min_alpha=0.025, min_count=0, workers=8)
    model.build_vocab(trainingSet)
    for epoch in range(10):
        print('Epoch: %s'%epoch)
        model.train(trainingSet, total_examples=model.corpus_count, epochs=model.epochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
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
    return mode, int(mean+0.5), median, int(sum(freq)*100/len(ranks)+0.5); 

def selfAssessment(train_corpus, model):
    for doc in train_corpus:
        inferred_vector = model.infer_vector(doc.words)
        yield doc.tags[0], cosineDistance(model["transit"], inferred_vector), cosineDistance(model["not_transit"], inferred_vector)

def cosineDistance(a,b):
    return dot(a, b)/(norm(a)*norm(b))

if __name__ == '__main__':    
    path = Path('cache/trainingSet.pickle')
    if path.exists():
        with path.open(mode='rb') as f:
            trainingSet = pickle.load(f)
    else:
        trainingSet = [td for td in loadTaggedDocuments(zip(*[i for i in loadTrainingSet('tweets/Classificados/')]))]
        with path.open(mode='wb') as f:
            pickle.dump(trainingSet,f)
    model = train(trainingSet, "cache/trainedModel.bin", False)
    
    #print("Mode: {stat[0]}, Mean: {stat[1]}, Median: {stat[2]}, Match: {stat[3]}%".format(stat=ranking(trainingSet, model)))
    res = collections.defaultdict(int)
    tr = 0
    for tag, t_cls, nt_cls in selfAssessment(trainingSet, model):
        if (tag=="transit") == (t_cls > nt_cls):
            res[tag] += 1
    for tag in res:
        print(tag, "{:.2f}%".format(res[tag]*200/len(trainingSet)))

    tweets = ["o transito na ipiranga esta lento", "o trafego na internet esta lento", "o trafego de rede esta lento"]
    for twt in tweets:
        tkns = [w for w in tokenizeTweet(twt)]
        iv = model.infer_vector(tkns)
        print(tkns)
        print(cosineDistance(model["transit"], iv))
        print(cosineDistance(model["not_transit"], iv))