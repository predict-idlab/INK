from ink.base.structure import InkExtractor
from ink.miner.rulemining import RuleSetMiner
from rdflib_hdt import HDTStore
from ink.base.connectors import RDFLibConnector, StardogConnector, AbstractConnector
from sklearn.metrics import accuracy_score
import sys
import pickle
import glob
from tqdm import tqdm
from rdflib import URIRef, Graph, Literal
###

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

    dir_kb = sys.argv[1]

    format = ''
    if sys.argv[2] == 'owl':
        format = 'xml'
    else:
        format=sys.argv[2]

    pos_file = sys.argv[3]
    neg_file = sys.argv[4]
    depth = int(sys.argv[5])

    print(dir_kb+"/*.hdt")
    file = list(glob.glob(dir_kb+"/*.hdt"))[0]

    connector = HDTConnector(file)
    #connector = RDFLibConnector(file, format)
    extractor = InkExtractor(connector, verbose=True)

    pos = set(line.strip() for line in open(pos_file))
    neg = set(line.strip() for line in open(neg_file))

    X_train, y_train = extractor.create_dataset(depth, pos, neg, jobs=1)
    X_train = extractor.fit_transform(X_train, counts=True, levels=True)

    extractor.connector.store=None
    pickle.dump(extractor, open(sys.argv[6]+'.extract','wb'))

    def func1(data):
        features = data[0].tocsc()
        cols = data[2]
        drops = set()
        for i in tqdm(range(0,features.shape[1])):
            #if 'real_data' not in cols[i]:
                if features[:,i].sum()==1 or features[:,i].getnnz()==1:
                    drops.add(i)
        n_cols = [j for j, i in tqdm(enumerate(cols)) if j not in drops]
        return features[:, n_cols], data[1], [i for j, i in enumerate(cols) if j not in drops]


    X_train = func1(X_train)


    model = RuleSetMiner(max_len_rule_set=3, forest_size=100, support=1, verbose=True)
    #model = rs.BayesianRuleSet()
    acc, rules = model.fit(X_train, y_train)

    pickle.dump(model, open(sys.argv[6],'wb'))
    model.print_rules(rules)
#print(res[0][1][0])
