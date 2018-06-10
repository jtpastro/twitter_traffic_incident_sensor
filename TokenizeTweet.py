from nltk.tokenize import TweetTokenizer
from nltk.stem import RSLPStemmer
from unidecode import unidecode as unicodeToAscii
from nltk.corpus import stopwords
import re

class Token(str):
    timeRegex = re.compile(r'^\d{1,2}((h(\d{2})?)|:\d{2})$')
    dateRegex = re.compile(r'^\d{1,2}/\d{1,2}(/\d{2,4})?$')
    wordRegex = re.compile(r'^([a-z-]+|[#@]\w+)$')
    numberRegex = re.compile(r'^-?\d+([,\.]?\d)*$')
    def __new__(cls, value):
        if Token.wordRegex.match(value):
            obj = str.__new__(cls, value)
        elif Token.timeRegex.match(value):
            obj = str.__new__(cls, '__TTKN__')
        elif Token.dateRegex.match(value):
            obj = str.__new__(cls, '__DTKN__')
        elif Token.numberRegex.match(value):
            obj = str.__new__(cls, '__NTKN__')
        else:
            obj = str.__new__(cls, '')
        obj.value = value
        return obj

class FilterTokenizer(TweetTokenizer):
    def __init__(self, filterStopwords=True, stemming=False, groupClasses=True):
        self.groupClasses = groupClasses
        if stemming:
            self.stemmer = RSLPStemmer()
        else:
            self.stemmer = lambda: None
            self.stemmer.stem = lambda x: x
        self.stopwords = [unicodeToAscii(sw) for sw in stopwords.words('portuguese')] if filterStopwords else []
        super().__init__(preserve_case=False,reduce_len=True)

    def tokenize(self, tweet):
        for tkn in super().tokenize(unicodeToAscii(tweet)):
            if self.groupClasses:
                tkn = Token(tkn)
            if len(tkn) > 1 and not tkn in self.stopwords:
                yield self.stemmer.stem(tkn)

if __name__ == '__main__':
    tt = FilterTokenizer()
    tweets = ["Problema mesmo \u00e9 na BR386: diria que h\u00e1 uns 10 km de congestionamento em cada sentido. \nNo C/I, tranca da BR448 at\u00e9 a Ponte do Ca\u00ed.\nNo I/C, tranca antes do acesso ao polo petroqu\u00edmico at\u00e9 a Ponte do Ca\u00ed.\nVai demorar algumas horas pra normalizar ap\u00f3s acidente @GauchaZH", "16h43 - Aproveite o final de semana com consci\u00eancia! \u00c1lcool e dire\u00e7\u00e3o n\u00e3o combinam. #Educa\u00e7\u00e3oEPTCpic.twitter.com/FAHeKL4UKF", "Fim das obras na Av. Crist\u00f3v\u00e3o Colombo com Ramiro Barcelos. Tr\u00e2nsito volta a fluir melhor na regi\u00e3o. Mas Ramiro segue movimentada na descida, rumo \u00e0 Legalidade @GauchaZH", "16h37 - Tr\u00e2nsito totalmente liberado na R. Ramiro Barcelos esq. com a Av. Crist\u00f3v\u00e3o Colombo. Tr\u00e2nsito fluindo bem na regi\u00e3o.", "Regi\u00e3o do Aeroporto bastante movimentada nesta tarde. Sa\u00edda com mais tr\u00e2nsito pela Terceira Perimetral e Sert\u00f3rio. Chegada \u00e0 Capital ainda sem tranqueiras @GauchaZHpic.twitter.com/uCD6lquRpP", " ATEN\u00c7\u00c3O PARA BLOQUEIO pic.twitter.com/1S8bokO7rq", " ATEN\u00c7\u00c3O PARA BLOQUEIO pic.twitter.com/IAttzhDSkU", "O curso EAD \"Pedalando com seguran\u00e7a\" gratuito\n\nInscri\u00e7\u00f5es: https://goo.gl/8aPAmJ\u00a0pic.twitter.com/S0nLHY7eBB", "ATEN\u00c7\u00c3O!!!!https://twitter.com/PRF191RS/status/972483823580647427\u00a0\u2026", "https://gauchazh.clicrbs.com.br/esportes/gauchao/noticia/2018/03/bm-reforca-policiamento-no-entorno-do-beira-rio-e-orienta-deslocamento-de-torcidas-para-o-gre-nal-413-cjelfllns01xs01p46a7ljlfu.html\u00a0\u2026", "BM refor\u00e7a policiamento no entorno do Beira-Rio e orienta deslocamento de torcidas para o Gre-Nal 413. O esquema de seguran\u00e7a, tr\u00e2nsito e locais das concentra\u00e7\u00f5es de torcidas aqui: \nhttps://gauchazh.clicrbs.com.br/esportes/gauchao/noticia/2018/03/bm-reforca-policiamento-no-entorno-do-beira-rio-e-orienta-deslocamento-de-torcidas-para-o-gre-nal-413-cjelfllns01xs01p46a7ljlfu.html\u00a0\u2026 @GauchaZHpic.twitter.com/tYjCSS3umr", "concentra\u00e7\u00e3o na pra\u00e7a do canh\u00e3o, na marinha. sa\u00edda \u00e0s 15h pro est\u00e1dio", "Segundo a EPTC, a tarifa da lota\u00e7\u00e3o pode variar no m\u00ednimo 1,4 vezes o valor do \u00f4nibus. Com a passagem de \u00f4nibus a 4,30, d\u00e1 6,02 em 1,4x. Foi arrendondado pra mais, 6,05, porque se ficasse em 6 a\u00ed seria menos de 1,4x. Entende? hehe", "Pois\u00e9, estamos esclarecendo isso agora. Obrigada pelo toque!", "a\u00ed tem que ver com as torcidas organizadas. s\u00e3o eles que organizam esses transportes.", "N\u00e3o h\u00e1 ciclofaixa? Ande do lado direito junto ao meio fio.pic.twitter.com/plQ5kMuVgU", "Usu\u00e1rios t\u00eam at\u00e9 segunda para recarregar cart\u00e3o TRI sem o reajuste da passagem de \u00f4nibus em Porto Alegre. Mais esclarecimentos sobre as mudan\u00e7as na tarifa em @GauchaZH: https://gauchazh.clicrbs.com.br/porto-alegre/noticia/2018/03/usuarios-tem-ate-segunda-para-recarregar-cartao-tri-sem-o-reajuste-da-passagem-de-onibus-em-porto-alegre-cjelddax701y101r4lbpirec4.html\u00a0\u2026 pic.twitter.com/th6keq4zOy",
                "a 10/12 1:2 c/i centro/bairro centro-bairro"]
    for tweet in tweets:
        print(" ".join([tkn for tkn in tt.tokenize(tweet)]))