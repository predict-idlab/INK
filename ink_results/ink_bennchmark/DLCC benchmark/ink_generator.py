from ink.base.connectors import AbstractConnector
from rdflib import URIRef, Graph, Literal
from rdflib_hdt import HDTStore
from ink.base.structure import InkExtractor
from tqdm import tqdm
from glob import glob


class HDTConnector(AbstractConnector):
    def query(self, q_str):
        global store
        try:
            noi = URIRef(q_str.split('"')[1])
            res = store.hdt_document.search((noi, None, None))[0]
            val = [{"p": {"value": r[1].toPython()}, "o": {"value": r[2].n3().split('"')[1]}, "dt": "Object"} if isinstance(r[2],
                                                                                                            Literal) else {
                "p": {"value": r[1].toPython()}, "o": {"value": r[2].toPython()}} for r in res]
            return val
        except Exception as e:
            return []

    def inv_query(self, q_str):
        global store
        try:
            noi = URIRef(q_str.split('"')[1])
            res = store.hdt_document.search((None, None, noi))[0]
            val = [{"p": {"value": "inv_"+r[1].toPython()}, "o": {"value": r[2].n3().split('"')[1]},
                    "dt": "Object"} if isinstance(r[2],
                                                  Literal) else {
                "p": {"value": "inv_"+r[1].toPython()}, "o": {"value": r[2].toPython()}} for r in res]
            return val
        except Exception as e:
            return []

    def get_all_events(self):
        global store
        res = store.hdt_document.search((None, None, None))[0]
        entities = set()
        for r in tqdm(res):
            entities.add(r[0].toPython())
        return entities



file = "/Users/bramsteenwinckel/Downloads/dl-evaluation-framework-master/v1/total.hdt"
store = HDTStore(file)
if __name__ == '__main__':
    connector = HDTConnector()
    extractor = InkExtractor(connector, extract_inverse=True, verbose=True)

    all_activities = {}

    for file in glob(
            '/Users/bramsteenwinckel/Downloads/dl-evaluation-framework-master/v1/synthetic_ontology/*/*/*/positives.txt'):
        with open(file, 'r') as f:
            if file.split('/')[-4] not in all_activities:
                all_activities[file.split('/')[-4]] = []
            all_activities[file.split('/')[-4]].extend(
                    ['http://example.com/' + d.replace('<', '').replace('>', '').replace('\n', '') for d in f.readlines()])
    for file in glob(
            '/Users/bramsteenwinckel/Downloads/dl-evaluation-framework-master/v1/synthetic_ontology/*/*/*/negatives.txt'):
        with open(file, 'r') as f:
            if file.split('/')[-4] not in all_activities:
                all_activities[file.split('/')[-4]] = []
            all_activities[file.split('/')[-4]].extend(
                ['http://example.com/' + d.replace('<', '').replace('>', '').replace('\n', '') for d in f.readlines()])

    pos = []
    for key in all_activities:
        pos += all_activities[key]
    # set(list(set([x['a']['value'] for x in results['results']['bindings']]))[0:10])#
    print(len(pos))
    X_train, y_train = extractor.create_dataset(1, pos, None, jobs=4,
                                                skip_list=[])

    X_train = extractor.fit_transform(X_train,counts=True,levels=True, float_rpr=False)
    # df_train = pd.DataFrame.sparse.from_spmatrix(X_train[0], index=X_train[1], columns=X_train[2])

    import pandas as pd

    def func1(data):
        features = data[0].tocsc()
        cols = data[2]
        drops = set()
        for i in tqdm(range(0, features.shape[1])):
            # if 'real_data' not in cols[i]:
            if features[:, i].sum() == 1 or features[:, i].getnnz() == 1:
                drops.add(i)
        n_cols = [j for j, i in tqdm(enumerate(cols)) if j not in drops]
        return features[:, n_cols], X_train[1], [i for j, i in enumerate(cols) if j not in drops]


    print(X_train[0].shape)
    if True:
        X_train = func1(X_train)
    print(X_train[0].shape)
    # df_train.to_csv('table.csv')
    df_train = pd.DataFrame.sparse.from_spmatrix(X_train[0], index=[x.split('/')[-1] for x in X_train[1]], columns=X_train[2])
    # print('start saving')
    df_train.to_pickle('ink_result.pkl')#.to_csv(r'ink_results.txt', header=None, sep=' ', chunksize=100000)