from pathlib import Path
from gensim.models.doc2vec import TaggedDocument
from TokenizeTweet import FilterTokenizer
from collections import defaultdict, Counter
from itertools import zip_longest, product, combinations, chain
from WordEmbeddingClassifier import Paragraph2VecClassifier, FastTextClassifier
import json
from time import process_time

def loadTrainingSet(initialPath):
    for path in Path(initialPath).iterdir():
        for tweetClass in zip_longest(*[(path3 for path3 in Path(path2).iterdir()) for path2 in Path(path).iterdir()]):
            for tweetFile in tweetClass:
                if tweetFile:
                    yield "test" if None in tweetClass else "train", tweetFile.parent.name, tweetFile

def loadFastTextFormat(ftFile, label="__label__"):
    tdList = []
    ftFile = Path(ftFile)
    if ftFile.exists():
        with ftFile.open(mode='r') as f:
            for line in f:
                tokens = line.split()
                tags = []
                words = []
                for token in tokens:
                    if label in token:
                        tags.append(token.split(label)[-1])
                    else:
                        words.append(token)
                    tdList.append(TaggedDocument(words, tags))
    return tdList

def saveFastTextFormat(inFiles, outFile, paramComb, label="__label__"):
    outFile = Path(outFile)
    tknzr = FilterTokenizer(**paramComb)
    with outFile.open(mode='w') as f:
        sId = 0
        for twtCls in inFiles:
            for tweetFile in inFiles[twtCls]:
                with Path(tweetFile).open(encoding="utf8") as f2:
                    text = f2.readline()
                sId += 1
                words = [tkn for tkn in tknzr.tokenize(text)]
                tokens = [" ".join(words), label + twtCls, label+str(sId)]
                f.write(" ".join(tokens)+'\n')

def generateParametersCombinationAll(params):
    for comb in product(*[product([pName], params[pName]) for pName in params]):
        yield dict(comb)

def generateParametersCombinationOneFixed(params):
    for pName in params:
        for param, value in product([pName], params[pName]):
            yield {param: value}

def printAsCSV(*args, out=None):
    if out:
        print(*args, sep=', ', file=out)
    print(*args, sep=', ')

def evaluate(params, classifier, trainingFile, testFile, outFile, iterations=10):
    print(outFile)
    with open(outFile,'w') as f:
        for j,paramComb in enumerate(generateParametersCombinationOneFixed(params)):
            ev = []
            timers = []
            for i in range(iterations):
                print("Iteration {:d}".format(i+1))
                start_time = process_time()
                cfier = classifier(trainingFile, **paramComb)
                timers.append(process_time() - start_time)
                ev.append(cfier.evaluate(testFile))
            evals = {l:{p: "{:.4}".format(sum(e[l][p] for e in ev)/len(ev)) for p in ev[0][l]} for l in ev[0]}
            if j == 0:
                pCounter = ('param'+str(i)+', val'+str(i) for i in range(len(paramComb)))
                printAsCSV(*pCounter,"class", *next(iter(evals.values())).keys(), "trainTime", out=f)
            for _cls in evals:
                params = (k+', '+str(v) for k,v in paramComb.items())
                printAsCSV(*params, _cls, *evals[_cls].values(), '{:.2f}'.format(round(sum(timers)/iterations, 2)), out=f)

def filterLines(file):
    with open(file) as f:
        while True:
            params = f.readline()
            result = f.readline()
            if not result:
                break
            if "'precision': (9" in result:
                print(params,result, end='')

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))

def loadBest(file, paramLen=1):
    return sorted([(k,c) for k,c in Counter(paramSubSet for params in json.load(open(file)) for paramSubSet in combinations(params.items(), paramLen)).most_common() if c > 1])

if __name__ == '__main__':
    tokenParameters = generateParametersCombinationAll({'filterStopwords':[False, True], 'stemming':[False, True], 'groupClasses':[False, True]})
    evalFiles = defaultdict(list)
    trainFileList = sorted(Path('input/').glob('train*.txt'))
    testFileList = sorted(Path('input/').glob('test*.txt'))
    if not any(trainFileList) or not any(testFileList):
        datasets = defaultdict(lambda: defaultdict(list))
        for setType,tweetClass,filePath in loadTrainingSet('tweets/Classificados/'):
            datasets[setType][tweetClass].append(filePath)
        for comb in tokenParameters:
            for setType in datasets:
                filename = setType+"_"+"_".join(("" if comb[k] else "not")+k for k in comb)
                evalFiles[setType].append("input/"+filename+".txt")
                saveFastTextFormat(datasets[setType], evalFiles[setType][-1], comb)   
    else:
        paramsP2V = {  
                    "vecSize": [25, 50, 100, 300],
                    "winSize": [3, 5, 9],
                    "epochs": [20, 40, 80],
                    "minCount": [0,1,5],
                    "lossFunction": [-5,-1,0,1],
                    "sampleThreshold": [0.01, 0.001, 0.0001],
                    "learnRate": [0.5, 0.25, 0.025, 0.0025],
                    "learnDropRate": [10/25, 1/25, 1/250],
                    "trainingAlgorithm": [0,1,2,3]
                }
        paramsFT = { 
                    "vecSize": [25,50,100,300],
                    "winSize": [3, 5, 9],
                    "epochs": [20, 40, 80],
                    "minCount": [0,1,5],
                    "lossFunction": [-5,-1,0,1],
                    "sampleThreshold": [0.01, 0.001, 0.0001],
                    "learnRate": [0.05, 0.1, 0.25, 0.5],
                    "ngrams": [0,5,6],
                    "wordGrams": [1,3,5],
                    "bucket": [0,1000000, 2000000, 10000000]
            }
        for trainFile, testFile in zip(trainFileList, testFileList):
            evaluate(paramsP2V, Paragraph2VecClassifier, trainFile, testFile, 'output/p2v'+trainFile.name.lstrip('train'))
            evaluate(paramsFT, FastTextClassifier, str(trainFile), str(testFile), 'output/ft'+trainFile.name.lstrip('train'))
               
        #filterLines("input/paragraph2vec_stats.txt")
        #for i in loadBest("filtered.txt"):
        #    print(*i)