import sys
import pickle
import math
from ink.base.structure import InkExtractor
from ink.base.connectors import RDFLibConnector
import glob
from tqdm import tqdm
from rdflib_hdt import HDTStore
from ink.base.connectors import RDFLibConnector, StardogConnector, AbstractConnector
from rdflib import URIRef, Graph, Literal
#########

#########
if __name__ == '__main__':
    class HDTConnector(AbstractConnector):
        def __init__(self, filename):
            self.filename = filename
            self.store = None

        def query(self, q_str):
            if self.store is None:
                self.store = HDTStore(self.filename)
            #if self.all_requests.check(q_str):
            #    return self.all_requests.get(q_str)
            try:
            # res = graph.query(q_str)
                noi = URIRef(q_str.split('"')[1])
                res = self.store.hdt_document.search((noi, None, None))[0]
                val = [{"p": {"value": r[1].toPython()}, "o": {"value": r[2].n3().split('"')[1]}} if isinstance(r[2],
                                                                                                                Literal) else {
                    "p": {"value": r[1].toPython()}, "o": {"value": r[2].toPython()}} for r in res]
                return val
            # return json.loads(res.serialize(format="json"))#['results']['bindings']
            except Exception as e:
                #    print(e)
                return []

    def perf_measure(y_actual, y_pred):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_pred)):
            if y_actual[i]==y_pred[i]==1:
               TP += 1
            if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
               FP += 1
            if y_actual[i]==y_pred[i]==0:
               TN += 1
            if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
               FN += 1

        return(TP, FP, TN, FN)

    ####

    dir_kb = sys.argv[1]

    format = ''
    if sys.argv[2] == 'owl':
        format = 'xml'
    else:
        format = sys.argv[2]

    pos_file = sys.argv[3]
    neg_file = sys.argv[4]
    depth = int(sys.argv[5])

    #print(glob.glob(dir_kb+"/*."+sys.argv[2]))
    #file = list(glob.glob(dir_kb+"/*."+sys.argv[2]))[0]

    extractor = pickle.load(open(sys.argv[6]+'.extract','rb'))
    #extractor = InkExtractor(connector, verbose=True)

    pos = set(line.strip() for line in open(pos_file))
    neg = set(line.strip() for line in open(neg_file))

    X_test, y_test = extractor.create_dataset(depth, pos, neg, jobs=1)
    X_test = extractor.transform(X_test, counts=True, levels=True)
    print(X_test)

    model = pickle.load(open(sys.argv[6], 'rb'))
    y_predict = model.predict(X_test)

    print("###")

    res = perf_measure(y_test, y_predict)
    TP = res[0]
    FP = res[1]
    TN = res[2]
    FN = res[3]
    try:
        mcc = (TP*TN - FP*FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    except:
        mcc = None
    print("#EVAL# tp:", res[0])
    print("#EVAL# fp:", res[1])
    print("#EVAL# tn:", res[2])
    print("#EVAL# fn:", res[3])
    print(mcc)
