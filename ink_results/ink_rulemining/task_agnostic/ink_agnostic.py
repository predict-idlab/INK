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
from timeit import default_timer as timer

class HDTConnector(AbstractConnector):

    def query(self, q_str):
        global store
        #if self.all_requests.check(q_str):
        #    return self.all_requests.get(q_str)
        try:
        # res = graph.query(q_str)
            noi = URIRef(q_str.split('"')[1])
            res = store.hdt_document.search((noi, None, None))[0]
            val = [{"p": {"value": r[1].toPython()}, "o": {"value": r[2].n3().split('"')[1]}} if isinstance(r[2],
                                                                                                            Literal) else {
                "p": {"value": r[1].toPython()}, "o": {"value": r[2].toPython()}} for r in res]
            return val
        # return json.loads(res.serialize(format="json"))#['results']['bindings']
        except Exception as e:
            #    print(e)
            return []

    def get_all_entities(self):
        global store
        res = store.hdt_document.search((None, None, None))[0]
        entities = set()
        for r in tqdm(res):
            entities.add(r[0].toPython())
            #entities.add(r[2].toPython())
        return entities


file = sys.argv[1]
store = HDTStore(file)

if __name__ == '__main__':    
    start = timer()
    connector = HDTConnector()
    extractor = InkExtractor(connector, verbose=True)
    pos = connector.get_all_entities()

    X_train, y_train = extractor.create_dataset(2, pos, None, jobs=32)
    X_train = extractor.fit_transform(X_train)
    end = timer()
    print("loading done in: ", str(end-start))
    model = RuleSetMiner(support=100, max_len_rule_set=3, verbose=True, rule_complexity=int(sys.argv[3]))
    print(X_train[0].shape)
    
    rf = model.fit(X_train)
    end2 = timer()
    print("mining done in:",str(end2-end))
    rf.to_csv(sys.argv[2]+'.csv')

    print("total time:",str(end2-start))
