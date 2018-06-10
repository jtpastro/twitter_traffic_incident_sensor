from pathlib import Path
from WordEmbeddingClassifier import Paragraph2VecClassifier, FastTextClassifier
from collections import defaultdict
from TrainTweets import printAsCSV
import inspect
import matplotlib.pyplot as plt

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def percentToDecimal(string):
    return '{:.4f}'.format(float(string.strip('%'))/100)

def convertFT(inFile, outFile):
    with Path(inFile).open() as f:
        with Path(outFile).open('w') as o:
            i=0
            while True:
                pNames = tuple(f.readline().strip('\n').split(', '))
                pValues = tuple(f.readline().strip('\n').split(', '))
                pr = tuple(f.readline().strip('\n').split(', '))
                tt = tuple(f.readline().strip('\n').split(', '))
                if tt[0]:
                    params = (i for param in zip(pNames, pValues) for i in param)
                    if i == 0: 
                        i+=1   
                        pCounter = ('param'+str(i)+', val'+str(i) for i in range(len(pNames)))               
                        printAsCSV(*pCounter, "class", "precision", "recall", "trainTime",out=o)
                    printAsCSV(*params, pr[0], percentToDecimal(pr[1]), percentToDecimal(pr[2]), tt[-1].strip('s'), out=o)
                else:
                    break      

def convertP2V(inFile, outFile):
    with Path(inFile).open() as f:
        with Path(outFile).open('w') as o:
            i=0
            while True:
                pNames = tuple(f.readline().strip('\n').split(', '))
                pValues = tuple(f.readline().strip('\n').split(', '))
                pr = tuple(f.readline().strip('\n').split(', '))
                pr2 = tuple(f.readline().strip('\n').split(', '))
                tt = tuple(f.readline().strip('\n').split(', '))
                if tt[0]:
                    params = [i for param in zip(pNames, pValues) for i in param]
                    if i == 0: 
                        i+=1   
                        pCounter = ('param'+str(i)+', val'+str(i) for i in range(len(pNames)))               
                        printAsCSV(*pCounter, "class", "precision", "recall", "trainTime",out=o)
                    printAsCSV(*params, pr[0], percentToDecimal(pr[1]), percentToDecimal(pr[2]), tt[-1].strip('s'), out=o)
                    printAsCSV(*params, pr2[0], percentToDecimal(pr2[1]), percentToDecimal(pr2[2]), tt[-1].strip('s'), out=o)
                else:
                    break    
def loadEvaluation(file):
    results = defaultdict(lambda: defaultdict(dict))
    with Path(file).open() as f:
        f.readline()
        while True:
            data = tuple(f.readline().strip('\n').split(', '))
            if data[0]:
                pName, pValue, _cls, precision, recall, duration = data
                results[pName][float(pValue)][_cls] = (round(float(precision), 2), round(float(recall),2), float(duration))
            else:
                break
    return results

            

if __name__ == '__main__':
    #for f in sorted(Path('output/').glob('p2v*.txt')):
    #    convertP2V(f, 'output/_'+f.name)
    resultsP2V = loadEvaluation(sorted(Path('output').glob('p2v*.txt'))[0])['vecSize']
    resultsFT = loadEvaluation(sorted(Path('output').glob('ft*.txt'))[0])['vecSize']
    x = resultsP2V.keys()
    y = tuple(zip(*(tuple(zip(*i)) for i in zip(*(k.values() for k in resultsP2V.values())))))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x,y[0][0], 'ro', label="precisão - trânsito")
    plt.plot(x,y[1][1], 'ro', label='revocação - não trânsito')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel('Tamanho do Vetor de Palavras')
    for i,j in zip(tuple(x)+tuple(x),y[0][0]+y[1][1]):
        ax.annotate(str((i,j)),xy=(i,j), xytext=(-8,-15), textcoords='offset points')
    #savefig("../figures/exercice_2.png",dpi=72)
    plt.show()
    