import requests
import json
from urllib import parse
from requests.adapters import HTTPAdapter
from requests import Session
from ink.base.connectors import AbstractConnector, StardogConnector
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from scipy import sparse
from tqdm import tqdm

def sparse_unique_columns(M):
    M = M.tocsc()
    m, n = M.shape
    if not M.has_sorted_indices:
        M.sort_indices()
    if not M.has_canonical_format:
        M.sum_duplicates()
    sizes = np.diff(M.indptr)
    idx = np.argsort(sizes)
    Ms = M@sparse.csc_matrix((np.ones((n,)), idx, np.arange(n+1)), (n, n))
    ssizes = np.diff(Ms.indptr)
    ssizes[1:] -= ssizes[:-1]
    grpidx, = np.where(ssizes)
    grpidx = np.concatenate([grpidx, [n]])
    if ssizes[0] == 0:
        counts = [np.array([0, grpidx[0]])]
    else:
        counts = [np.zeros((1,), int)]
    ssizes = ssizes[grpidx[:-1]].cumsum()
    for i, ss in tqdm(enumerate(ssizes)):
        gil, gir = grpidx[i:i+2]
        pl, pr = Ms.indptr[[gil, gir]]
        dv = Ms.data[pl:pr].view(f'V{ss*Ms.data.dtype.itemsize}')
        iv = Ms.indices[pl:pr].view(f'V{ss*Ms.indices.dtype.itemsize}')
        idxi = np.lexsort((dv, iv))
        dv = dv[idxi]
        iv = iv[idxi]
        chng, = np.where(np.concatenate(
            [[True], (dv[1:] != dv[:-1]) | (iv[1:] != iv[:-1]), [True]]))
        counts.append(np.diff(chng))
        idx[gil:gir] = idx[gil:gir][idxi]
    counts = np.concatenate(counts)
    nu = counts.size - 1
    uniques = M@sparse.csc_matrix((np.ones((nu,)), idx[counts[:-1].cumsum()],
                                   np.arange(nu + 1)), (n, nu))
    return uniques, idx, counts[1:]

class EndpointConnector(AbstractConnector):
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.session = Session()
        adapter = HTTPAdapter(pool_connections=10000, pool_maxsize=10000)
        self.session.mount('http://', adapter)

    def query(self, q_str):
        query = parse.quote(q_str)
        r = self.session.get(self.endpoint + '?query=' + query,
                             headers={"Accept": "application/sparql-results+json"},  timeout=1).text
        return json.loads(r)['results']['bindings']

if __name__ == '__main__':
    df = pd.read_csv('movies/completeDataset.tsv', delimiter='\t')
    df_train = pd.read_csv('movies/TrainingSet.csv', delimiter='\t').dropna()
    df_test = pd.read_csv('movies/TestSet.csv', delimiter='\t').dropna()

    urls = df['DBpedia_URI'].values


    #$$dbpedia = "http://10.2.32.192:8890/sparql"#"https://dbpedia.org/sparql" #"http://10.2.32.192:8890/sparql"#
    dbpedia = "https://dbpedia.org/sparql"
    con_details={
        'endpoint': 'http://10.2.35.70:5820'
    }
    connector = StardogConnector(con_details,"dbpedia",False)

    from ink.base.structure import InkExtractor
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import GaussianNB
    from ink.miner.rulemining import RuleSetMiner

    prefs = {'http://dbpedia.org/ontology':'dbo','http://dbpedia.org/resource':'dbr', 'http://www.w3.org/2000/01/rdf-schema':'rdfs',
             'http://www.w3.org/1999/02/22-rdf-syntax-ns':'rdf', 'http://www.w3.org/2002/07/owl':'owl'}
    extractor = InkExtractor(connector, prefixes=prefs,verbose=True)

    pos = set(urls)

    skip = ["http://dbpedia.org/ontology/abstract"]

    #s=""
    #for p in pos:
    #    s+="<"+p+"> "
    #print(s)
    #exit(0)

    X_train, _ = extractor.create_dataset(1,pos,skip_list=skip, jobs=4)
    X_train = extractor.fit_transform(X_train, counts=False, levels=False)

    #with open('movies/depth_1.p', 'wb') as file:
    #    pickle.dump(X_train, file)

    from sklearn.preprocessing import MaxAbsScaler
    scale = MaxAbsScaler()
    data = scale.fit_transform(X_train[0])
    data.data = np.nan_to_num(data.data, copy=False)
    main_inds = X_train[1]

    print(data.shape)
#    cols = X_train[2]



    train_inds = [main_inds.index(i) for i in df_train['DBpedia_URI'].values]
    train_data = data[train_inds,:]
    train_labels = df_train['Label'].values

    test_inds = [main_inds.index(i) for i in df_test['DBpedia_URI'].values]
    test_data = data[test_inds, :]
    test_labels = df_test['Label'].values


    #df_data = pd.DataFrame.sparse.from_spmatrix(data)
    #df_data.columns = X_train[2]
    #df_data.index = [x for x in X_train[1]]
    #df_data.columns = X_train[2]
    #print(df_data)
    #df_data = df_data.loc[:, ~df_data.T.duplicated(keep='first')]


    from sklearn.ensemble import ExtraTreesClassifier

    clf = ExtraTreesClassifier(n_estimators=500, random_state = 42)#MultinomialNB(alpha=1)#KNeighborsClassifier(n_neighbors=30)#ExtraTreesClassifier(n_estimators=100, random_state = 42)
    clf.fit(train_data, train_labels)

    print(accuracy_score(test_labels, clf.predict(test_data)))



