from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from ink.base.structure import InkExtractor
from ink.base.connectors import AbstractConnector
from rdflib_hdt import HDTStore
from sklearn import preprocessing
import pandas as pd
from rdflib import URIRef

store = HDTStore("data_node_class/AIFB/aifb.hdt")
class HDTConnector(AbstractConnector):
    def __init__(self, filename):
        self.filename = filename

    def query(self, q_str):
        """
        Execute a query through RDFLib
        :param q_str: Query string.
        :type q_str: str
        :return: Dictionary generated from the ['results']['bindings'] json.
        :rtype: dict
        """

        if isinstance(q_str, int):
            id = q_str
        else:
            id = store.hdt_document.term_to_id(URIRef(q_str), 0)

        res = store.hdt_document.search_ids((id, None, None))
        return res[0]

    def skip_ids(self, skip_lst):
        return [store.hdt_document.term_to_id(URIRef(s), e) for s, e in skip_lst]

    def get_rdf(self, var, kind):
        return store.hdt_document.id_to_term(var, kind)

if __name__ == '__main__':

    connector = HDTConnector("data_node_class/AIFB/aifb.hdt")

    df_train = pd.read_csv("data_node_class/AIFB/AIFB_train.tsv", delimiter='\t')
    df_test = pd.read_csv("data_node_class/AIFB/AIFB_test.tsv", delimiter='\t')

    data = pd.concat([df_train, df_test])

    le = preprocessing.LabelEncoder()
    df_train['label'] = le.fit_transform(df_train["label_affiliation"])
    df_test['label'] = le.transform(df_test["label_affiliation"])

    pos_file = set(['<' + x + '>' for x in data["person"].values])

    skip = connector.skip_ids([('http://swrc.ontoware.org/ontology#employs',1), ('http://swrc.ontoware.org/ontology#affiliation',1)])
    print(skip)

    extractor = InkExtractor(connector, verbose=True)
    X_train, _ = extractor.create_dataset(5, pos_file, set() ,skip_list=skip, jobs=4)
    extracted_data = extractor.fit_transform(X_train, counts=False, levels=False, float_rpr=False)

    inds_train =  [[x[1:-1] for x in extracted_data[1]].index(x) for x in df_train["person"].values]
    inds_test =  [[x[1:-1] for x in extracted_data[1]].index(x) for x in df_test["person"].values]


    X = extracted_data[0][inds_train,:]
    y = df_train['label'].values

    X_test = extracted_data[0][inds_test,:]
    y_test = df_test['label'].values

    clf_1 = ExtraTreesClassifier(n_estimators=100)

    clf_1.fit(X, y)

    y_pred_1 = clf_1.predict(X_test)

    print(accuracy_score(y_test, y_pred_1))