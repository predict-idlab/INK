from tqdm import tqdm
import pandas as pd
import numpy as np
from ink.base.structure import InkExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from ink.base.structure import InkExtractor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from ink.miner.rulemining import RuleSetMiner
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from ink.base.connectors import AbstractConnector, StardogConnector
import pickle
from pyrdf2vec.graphs import KG
import pandas as pd
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from multiprocessing import Pool
from hashlib import md5
from typing import List,Set, Tuple, Any
from tqdm import tqdm
import rdflib
import os, csv

class MultiProcessingRandomWalker(RandomWalker):
    def _proc(self, t):
        kg, instance = t
        walks = self.extract_random_walks(kg, instance)
        canonical_walks = set()
        for walk in walks:
            canonical_walk = []
            for i, hop in enumerate(walk):  # type: ignore
                if i == 0 or i % 2 == 1:
                    canonical_walk.append(str(hop))
                else:
                    digest = md5(str(hop).encode()).digest()[:8]
                    canonical_walk.append(str(digest))
            canonical_walks.add(tuple(canonical_walk))

        return {instance:tuple(canonical_walks)}

    #overwrite this method
    def _extract(self, kg: KG, instances: List[rdflib.URIRef]) -> Set[Tuple[Any, ...]]:
        canonical_walks = set()
        seq = [(kg, r) for _,r in enumerate(instances)]
        #print(self.depth)
        with Pool(8) as pool:
            res = list(tqdm(pool.imap_unordered(self._proc, seq), total=len(seq)))
        res = {k:v for element in res for k,v in element.items()}
        for r in instances:
            canonical_walks.update(res[r])

        return canonical_walks
###


class StreamConnector(AbstractConnector):
    def __init__(self, data):
        self.data = data

    def query(self, q_str):
        noi = q_str.split('"')[1]
        lst = []

        if noi in data:
            for key in self.data[noi]:
                for val in self.data[noi][key]:
                    val = val.split('^^')
                    if len(val)>1:
                        lst.append({'p': {'value': key}, 'o': {'value': val[0].replace('"','')}, 'dt': {'value': val[1]}})
                    else:
                        lst.append({'p': {'value': key}, 'o': {'value': val[0]}})

        #r = self.session.get(self.endpoint + '?query=' + query,
        #                     headers={"Accept": "application/sparql-results+json"},  timeout=1).text
        return lst
        #return json.loads(r)['results']['bindings']

data = {}

with open('/Users/bramsteenwinckel/Downloads/19/normal_obs.pk', 'rb') as file:
    all = set(pickle.load(file))
#normal = list(pd.read_csv('/Users/bramsteenwinckel/Downloads/19/molding_normal.csv')['s'].values)#[:300]

with open('/Users/bramsteenwinckel/Downloads/19/anomaly_obs.pk', 'rb') as file:
    anomalies = set(pickle.load(file))
#normal = list(pd.read_csv('/Users/bramsteenwinckel/Downloads/19/molding_normal.csv')['s'].values)#[:300]

normal = all-anomalies

#anomalies = list(pd.read_csv('/Users/bramsteenwinckel/Downloads/19/molding_anomaly.csv')['s'].values)#[:300]

#########################
if __name__ == '__main__':

######################@@



    # with open('/Users/bramsteenwinckel/Downloads/19/molding_machine_5000dp.metadata.nt') as f:
    #     lines = f.readlines()
    #     for line in tqdm(lines):
    #         spo = line.replace('<','').replace('>','').split(' ')
    #         s = spo[0]
    #         p = spo[1]
    #         o = spo[2]
    #
    #         if s not in data:
    #             data[s] = {}
    #         if p not in data[s]:
    #             data[s][p] = []
    #
    #         data[s][p].append(o)

    # with open('/Users/bramsteenwinckel/Downloads/19/reverse_relation.nt') as f:
    #     lines = f.readlines()
    #     for line in tqdm(lines):
    #         spo = line.replace('<', '').replace('>', '').split(' ')
    #         s = spo[0]
    #         p = spo[1]
    #         o = spo[2]
    #
    #         if s not in data:
    #             data[s] = {}
    #         if p not in data[s]:
    #             data[s][p] = []
    #
    #         data[s][p].append(o)



    # total_parts = {}
    # def _define_neighborhood(noi, prev_key, depth):
    #     if noi in data:
    #         for key in data[noi]:
    #             if prev_key != '':
    #                 if prev_key+'.'+key not in total_parts:
    #                     total_parts[prev_key+'.'+key] = []
    #                 total_parts[prev_key+'.'+key].extend([x for x in data[noi][key] if not x.startswith('_')])
    #             else:
    #                 if key not in total_parts:
    #                     total_parts[key] = []
    #                 total_parts[key].extend([x for x in data[noi][key] if not x.startswith('_')])
    #
    #             #next
    #             if depth-1>0:
    #                 for obj in data[noi][key]:
    #                     if prev_key != '':
    #                         total_parts.update(_define_neighborhood(obj, prev_key+'.'+key, depth-1))
    #                     else:
    #                         total_parts.update(_define_neighborhood(obj, key, depth - 1))
    #     return total_parts
    #
    #
    # d_data = []
    # for noi in tqdm(normal+anomalies):
    #     total_parts = {}
    #     d_data.append((noi, _define_neighborhood('<'+noi+'>','',5)))

    depth = 1
    details = {'endpoint': 'http://localhost:5820'}
    skf = StratifiedKFold(n_splits=10)
    from imblearn.under_sampling import NearMiss

    technique = 'RDF2Vec'
    from sklearn.feature_selection import VarianceThreshold
    if technique=='INK':
        with open('/Users/bramsteenwinckel/Downloads/19/molding_machine_5000dp.nt') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                spo = line.replace('<', '').replace('>', '').split(' ')
                s = spo[0]
                p = spo[1]
                o = spo[2]

                if s not in data:
                    data[s] = {}
                if p not in data[s]:
                    data[s][p] = []

                data[s][p].append(o)

        with open('/Users/bramsteenwinckel/Downloads/19/previous_relation.nt') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                spo = line.replace('<', '').replace('>', '').split(' ')
                s = spo[0]
                p = spo[1]
                o = spo[2]

                if s not in data:
                    data[s] = {}
                if p not in data[s]:
                    data[s][p] = []

                data[s][p].append(o)


        connector = StreamConnector(data)#StardogConnector(details, "test_anomaly")
        extractor = InkExtractor(connector, verbose=True)
        d_data,labels = extractor.create_dataset(depth,set(anomalies), set(normal), skip_list=['http://www.agtinternational.com/ontologies/I4.0#contains'])
        d_data = extractor.fit_transform(d_data, counts=False, levels=False)




        from sklearn.utils import resample
        from imblearn.under_sampling import RandomUnderSampler


        print(d_data[0].shape)
        v = VarianceThreshold(threshold=0.001)
        var_data = v.fit_transform(d_data[0])
        print(var_data.shape)

        #var_data = d_data[0]

        #res = clf.fit_predict(d_data[0])
        #print(res)
        #exit(0)
    else:
        kg = KG(location="http://localhost:5820/test_anomaly/query", is_remote=True,
                label_predicates=['http://www.agtinternational.com/ontologies/I4.0#contains'])
        walkers = [RandomWalker(1, None)]
        embedder = Word2Vec(size=10, sg=1)
        transformer = RDF2VecTransformer(walkers=walkers, embedder=embedder)
        inds = list(normal.union(anomalies))
        embeddings = transformer.fit_transform(kg, inds)

        labels = np.array([1 if x in anomalies else 0 for x in inds])
        var_data = np.array(embeddings)


    if not os.path.exists('results.csv'):
        with open('results.csv', 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'depth', 'F1', 'Bal_acc', 'conf'])

    for train_index, test_index in skf.split(var_data, labels):
        X_train, X_test = var_data[train_index], var_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        rus = NearMiss(version=2)

        X_train, y_train = rus.fit_resample(X_train,y_train)

        clf_6 = ExtraTreesClassifier(n_estimators=100)
        clf_7 = RandomForestClassifier(n_estimators=100)


        clf_6.fit(X_train, y_train)
        print(f1_score(y_test, clf_6.predict(X_test)))

        with open('results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Extra', depth, f1_score(y_test, clf_6.predict(X_test)),
                 balanced_accuracy_score(y_test, clf_6.predict(X_test)),
                 confusion_matrix(y_test, clf_6.predict(X_test))])

        clf_7.fit(X_train, y_train)
        with open('results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Random', depth, f1_score(y_test, clf_7.predict(X_test)),
                 balanced_accuracy_score(y_test, clf_7.predict(X_test)),
                 confusion_matrix(y_test, clf_7.predict(X_test))])

