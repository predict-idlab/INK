
import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
import numpy as np

import sys
import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from ink.base.connectors import StardogConnector
from ink.base.structure import InkExtractor
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.feature_selection import VarianceThreshold

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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from pympler import asizeof

###
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
        with Pool(4) as pool:
            res = list(pool.imap_unordered(self._proc, seq))
        res = {k:v for element in res for k,v in element.items()}
        for r in instances:
            canonical_walks.update(res[r])

        return canonical_walks
###
import os.path
import csv

if not os.path.exists('results.csv'):
    with open('results.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Method', 'depth', 'NB_acc', 'NN_acc', 'DT_acc', 'SVC_acc', 'LR_acc', 'Extra_acc', 'Random_acc',
         'NB_weighted_F1', 'NN_weighted_F1', 'DT_weighted_F1', 'SVC_weighted_F1', 'LR_weighted_F1', 'Extra_weighted_F1', 'Random_weighted_F1',
         'NB_weighted_precision', 'NN_weighted_precision', 'DT_weighted_precision', 'SVC_weighted_precision', 'LR_weighted_precision', 'Extra_weighted_precision',
         'Random_weighted_precision', 'NB_weighted_recall', 'NN_weighted_recall', 'DT_weighted_recall', 'SVC_weighted_recall',
         'LR_weighted_recall', 'Extra_weighted_recall','Random_weighted_recall',
         'Create_time', 'NB_Train_time','NN_Train_time','DT_Train_time','SVC_Train_time','LR_Train_time','Extra_Train_time','Random_Train_time',
         'NB_Test_time', 'NN_Test_time', 'DT_Test_time', 'SVC_Test_time', 'LR_Test_time', 'Extra_Test_time','Random_Test_time',
         'Memory'])


""" parameters """
if __name__ == "__main__":

    rel = False

    dataset = sys.argv[1]#'BGS'#'BGS'
    depth = int(sys.argv[2])
    method = sys.argv[3]


    dir_kb = '../data_node_class/'+dataset
    files = {'AIFB':'aifb.n3','BGS':'BGS.nt','MUTAG':'mutag.owl','AM':'rdf_am-data.ttl'}
    file = files[dataset]#'AIFB.n3'#'rdf_am-data.ttl'

    formats = {'AIFB':'n3','BGS':'nt','MUTAG':'owl','AM':'ttl'}

    format = formats[dataset]

    train = '../data_node_class/'+dataset+'/'+dataset+'_train.tsv'
    test = '../data_node_class/'+dataset+'/'+dataset+'_test.tsv'
    #train = 'mela/train.csv'
    #test = 'mela_tes'

    excludes_dict = {'AIFB':['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation'],'BGS':['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis'],'MUTAG':['http://dl-learner.org/carcinogenesis#isMutagenic'],'AM':['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material']}

    excludes = excludes_dict[dataset]#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']#['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']

    labels_dict = {'AIFB':'label_affiliation','BGS':'label_lithogenesis','MUTAG':'label_mutagenic','AM':'label_cateogory'}
    label_name = labels_dict[dataset]#'label_lithogenesis'#'label_affiliation'#'label_cateogory'#'label_lithogenesis'#'label_mutagenic'#'label_affiliation'

    items_dict = {'AIFB':'person','BGS':'rock','MUTAG':'bond','AM':'proxy'}
    items_name = items_dict[dataset]#'rock'#'person'#'proxy'#'rock'#'bond'#'person'

    #pos_file = 'mela/pos_mela.txt'
    #neg_file = 'mela/neg_mela.txt'

    df_train = pd.read_csv(train, delimiter='\t')
    df_test = pd.read_csv(test, delimiter='\t')

    data = pd.concat([df_train, df_test])

    le = preprocessing.LabelEncoder()
    df_train['label'] = le.fit_transform(df_train[label_name])
    df_test['label'] = le.transform(df_test[label_name])

    #print(df_train['label'].value_counts())
    #print(df_test['label'].value_counts())

    pos_file = set(['<' + x + '>' for x in data[items_name].values])


    ink_total_NB = []
    ink_total_NN = []
    ink_total_tree = []
    ink_total_support = []
    ink_total_log = []
    ink_total_extra = []
    ink_total_random = []

    ink_time_create = []
    ink_time_train = []
    ink_time_test = []

    ink_memory = []


    rdf_total_NB = []
    rdf_total_NN = []
    rdf_total_tree = []
    rdf_total_support = []
    rdf_total_log = []
    rdf_total_extra = []
    rdf_total_random = []

    rdf_time_create = []
    rdf_time_train = []
    rdf_time_test = []

    rdf_memory = []

    details = {'endpoint': 'http://localhost:5820'}
    connector = StardogConnector(details, dataset)
    connector.upload_kg(dir_kb+'/'+file)

    #for _ in tqdm(range(5)):

    ## INK exrtact

    if method=='INK':
        t0 = time.time()
        extractor = InkExtractor(connector, verbose=False)
        X_train, y_train = extractor.create_dataset(depth, pos_file, set(), excludes, jobs=4)
        extracted_data = extractor.fit_transform(X_train, counts=False, levels=False)

        df_data = pd.DataFrame.sparse.from_spmatrix(extracted_data[0])
        df_data.index = [x[1:-1] for x in extracted_data[1]]
        df_data.columns = extracted_data[2]

        threshold_n = 0.75
        sel = VarianceThreshold(threshold=(threshold_n * (1 - threshold_n)))
        sel_var = sel.fit_transform(df_data)
        df_data = df_data[df_data.columns[sel.get_support(indices=True)]]

        ink_time_create.append(time.time()-t0)

        ink_memory.append(asizeof.asizeof(df_data))



        df_train_extr = df_data[df_data.index.isin(df_train[items_name].values)]  # df_data.loc[[df_train['proxy']],:]
        df_test_extr = df_data[df_data.index.isin(df_test[items_name].values)]  # df_data.loc[[df_test['proxy']],:]

        df_train_extr = df_train_extr.merge(df_train[[items_name, 'label']], left_index=True, right_on=items_name)
        df_test_extr = df_test_extr.merge(df_test[[items_name, 'label']], left_index=True, right_on=items_name)

        ####
        X = df_train_extr.drop(['label', items_name], axis=1).values
        y = df_train_extr['label'].values

        clf_1 = KNeighborsClassifier(n_neighbors=3)
        clf_2 = GaussianNB()
        clf_3 = DecisionTreeClassifier()
        clf_4 = GridSearchCV(SVC(), {'C': [10 ** -3, 10 ** -2, 0.1, 1, 10, 10 ** 2, 10 ** 3]}, cv=3, n_jobs=4)
        clf_5 = GridSearchCV(LogisticRegression(),
                             {'C': [10 ** -3, 10 ** -2, 0.1, 1, 10, 10 ** 2, 10 ** 3], 'max_iter': [10000]}, cv=3,
                             n_jobs=4)
        clf_6 = ExtraTreesClassifier(n_estimators=100)
        clf_7 = RandomForestClassifier(n_estimators=100)

        # INK train:
        t1 = time.time()
        clf_1.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_2.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_3.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_4.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_5.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_6.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_7.fit(X, y)
        ink_time_train.append(time.time() - t1)


        # INK predict
        t2 = time.time()
        y_pred_1 = clf_1.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_2 = clf_2.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_3 = clf_3.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_4 = clf_4.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_5 = clf_5.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_6 = clf_6.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_7 = clf_7.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)

        ink_total_NN.append(accuracy_score(df_test_extr['label'].values, y_pred_1))
        ink_total_NN.append(f1_score(df_test_extr['label'].values, y_pred_1, average='weighted'))
        ink_total_NN.append(precision_score(df_test_extr['label'].values, y_pred_1, average='weighted'))
        ink_total_NN.append(recall_score(df_test_extr['label'].values, y_pred_1, average='weighted'))

        ink_total_NB.append(accuracy_score(df_test_extr['label'].values, y_pred_2))
        ink_total_NB.append(f1_score(df_test_extr['label'].values, y_pred_2, average='weighted'))
        ink_total_NB.append(precision_score(df_test_extr['label'].values, y_pred_2, average='weighted'))
        ink_total_NB.append(recall_score(df_test_extr['label'].values, y_pred_2, average='weighted'))

        ink_total_tree.append(accuracy_score(df_test_extr['label'].values, y_pred_3))
        ink_total_tree.append(f1_score(df_test_extr['label'].values, y_pred_3, average='weighted'))
        ink_total_tree.append(precision_score(df_test_extr['label'].values, y_pred_3, average='weighted'))
        ink_total_tree.append(recall_score(df_test_extr['label'].values, y_pred_3, average='weighted'))

        ink_total_support.append(accuracy_score(df_test_extr['label'].values, y_pred_4))
        ink_total_support.append(f1_score(df_test_extr['label'].values, y_pred_4, average='weighted'))
        ink_total_support.append(precision_score(df_test_extr['label'].values, y_pred_4, average='weighted'))
        ink_total_support.append(recall_score(df_test_extr['label'].values, y_pred_4, average='weighted'))

        ink_total_log.append(accuracy_score(df_test_extr['label'].values, y_pred_5))
        ink_total_log.append(f1_score(df_test_extr['label'].values, y_pred_5, average='weighted'))
        ink_total_log.append(precision_score(df_test_extr['label'].values, y_pred_5, average='weighted'))
        ink_total_log.append(recall_score(df_test_extr['label'].values, y_pred_5, average='weighted'))

        ink_total_extra.append(accuracy_score(df_test_extr['label'].values, y_pred_6))
        ink_total_extra.append(f1_score(df_test_extr['label'].values, y_pred_6, average='weighted'))
        ink_total_extra.append(precision_score(df_test_extr['label'].values, y_pred_6, average='weighted'))
        ink_total_extra.append(recall_score(df_test_extr['label'].values, y_pred_6, average='weighted'))

        ink_total_random.append(accuracy_score(df_test_extr['label'].values, y_pred_7))
        ink_total_random.append(f1_score(df_test_extr['label'].values, y_pred_7, average='weighted'))
        ink_total_random.append(precision_score(df_test_extr['label'].values, y_pred_7, average='weighted'))
        ink_total_random.append(recall_score(df_test_extr['label'].values, y_pred_7, average='weighted'))

        #Store
        with open('results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [dataset, 'INK', depth, ink_total_NB[0], ink_total_NN[0], ink_total_tree[0], ink_total_support[0],
                 ink_total_log[0], ink_total_extra[0], ink_total_random[0],ink_total_NB[1], ink_total_NN[1], ink_total_tree[1], ink_total_support[1],
                 ink_total_log[1], ink_total_extra[1], ink_total_random[1],ink_total_NB[2], ink_total_NN[2], ink_total_tree[2], ink_total_support[2],
                 ink_total_log[2], ink_total_extra[2], ink_total_random[2],ink_total_NB[3], ink_total_NN[3], ink_total_tree[3], ink_total_support[3],
                 ink_total_log[3], ink_total_extra[3], ink_total_random[3], ink_time_create[0], ink_time_train[0],ink_time_train[1],ink_time_train[2],ink_time_train[3],ink_time_train[4],ink_time_train[5], ink_time_train[6],
                 ink_time_test[0], ink_time_test[1], ink_time_test[2], ink_time_test[3], ink_time_test[4], ink_time_test[5], ink_time_test[6], ink_memory[0]])

    ## RDF2Vec extract:
    if method=='RDF2Vec':
        # extract
        t0 = time.time()
        kg = KG(location="http://localhost:5820/"+str(dataset)+"/query", is_remote=True, label_predicates=excludes)
        walkers = [MultiProcessingRandomWalker(depth, 500, UniformSampler())]
        embedder = Word2Vec(size=500, sg=1)
        transformer = RDF2VecTransformer(walkers=walkers, embedder=embedder)
        inds = [ind[1:-1] for ind in list(pos_file)]
        embeddings = transformer.fit_transform(kg, inds)
        rdf_time_create.append(time.time()-t0)

        rdf_memory.append(asizeof.asizeof(embeddings))

        #Train
        t1 = time.time()
        train_inds = [inds.index(v) for v in df_train[items_name].values]
        test_inds = [inds.index(v) for v in df_test[items_name].values]

        X = [embeddings[i] for i in train_inds]
        y = df_train['label'].values

        clf_1 = KNeighborsClassifier(n_neighbors=3)
        clf_2 = GaussianNB()
        clf_3 = DecisionTreeClassifier()
        clf_4 = GridSearchCV(SVC(), {'C': [10 ** -3, 10 ** -2, 0.1, 1, 10, 10 ** 2, 10 ** 3]}, cv=3, n_jobs=4)
        clf_5 = GridSearchCV(LogisticRegression(),
                             {'C': [10 ** -3, 10 ** -2, 0.1, 1, 10, 10 ** 2, 10 ** 3], 'max_iter': [10000]}, cv=3,
                             n_jobs=4)
        clf_6 = ExtraTreesClassifier(n_estimators=100)
        clf_7 = RandomForestClassifier(n_estimators=100)

        # INK train:
        t1 = time.time()
        clf_1.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_2.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_3.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_4.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_5.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_6.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_7.fit(X, y)
        rdf_time_train.append(time.time() - t1)

        # predict

        t2 = time.time()
        y_pred_1 = clf_1.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_2 = clf_2.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_3 = clf_3.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_4 = clf_4.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_5 = clf_5.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_6 = clf_6.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_7 = clf_7.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)

        rdf_total_NN.append(accuracy_score(df_test['label'].values, y_pred_1))
        rdf_total_NN.append(f1_score(df_test['label'].values, y_pred_1, average='weighted'))
        rdf_total_NN.append(precision_score(df_test['label'].values, y_pred_1, average='weighted'))
        rdf_total_NN.append(recall_score(df_test['label'].values, y_pred_1, average='weighted'))

        rdf_total_NB.append(accuracy_score(df_test['label'].values, y_pred_2))
        rdf_total_NB.append(f1_score(df_test['label'].values, y_pred_2, average='weighted'))
        rdf_total_NB.append(precision_score(df_test['label'].values, y_pred_2, average='weighted'))
        rdf_total_NB.append(recall_score(df_test['label'].values, y_pred_2, average='weighted'))

        rdf_total_tree.append(accuracy_score(df_test['label'].values, y_pred_3))
        rdf_total_tree.append(f1_score(df_test['label'].values, y_pred_3, average='weighted'))
        rdf_total_tree.append(precision_score(df_test['label'].values, y_pred_3, average='weighted'))
        rdf_total_tree.append(recall_score(df_test['label'].values, y_pred_3, average='weighted'))

        rdf_total_support.append(accuracy_score(df_test['label'].values, y_pred_4))
        rdf_total_support.append(f1_score(df_test['label'].values, y_pred_4, average='weighted'))
        rdf_total_support.append(precision_score(df_test['label'].values, y_pred_4, average='weighted'))
        rdf_total_support.append(recall_score(df_test['label'].values, y_pred_4, average='weighted'))

        rdf_total_log.append(accuracy_score(df_test['label'].values, y_pred_5))
        rdf_total_log.append(f1_score(df_test['label'].values, y_pred_5, average='weighted'))
        rdf_total_log.append(precision_score(df_test['label'].values, y_pred_5, average='weighted'))
        rdf_total_log.append(recall_score(df_test['label'].values, y_pred_5, average='weighted'))

        rdf_total_extra.append(accuracy_score(df_test['label'].values, y_pred_6))
        rdf_total_extra.append(f1_score(df_test['label'].values, y_pred_6, average='weighted'))
        rdf_total_extra.append(precision_score(df_test['label'].values, y_pred_6, average='weighted'))
        rdf_total_extra.append(recall_score(df_test['label'].values, y_pred_6, average='weighted'))

        rdf_total_random.append(accuracy_score(df_test['label'].values, y_pred_7))
        rdf_total_random.append(f1_score(df_test['label'].values, y_pred_7, average='weighted'))
        rdf_total_random.append(precision_score(df_test['label'].values, y_pred_7, average='weighted'))
        rdf_total_random.append(recall_score(df_test['label'].values, y_pred_7, average='weighted'))

        # store
        with open('results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [dataset, 'RDF2Vec', depth, rdf_total_NB[0], rdf_total_NN[0], rdf_total_tree[0], rdf_total_support[0],
                 rdf_total_log[0], rdf_total_extra[0], rdf_total_random[0], rdf_total_NB[1], rdf_total_NN[1],
                 rdf_total_tree[1], rdf_total_support[1],
                 rdf_total_log[1], rdf_total_extra[1], rdf_total_random[1], rdf_total_NB[2], rdf_total_NN[2],
                 rdf_total_tree[2], rdf_total_support[2],
                 rdf_total_log[2], rdf_total_extra[2], rdf_total_random[2], rdf_total_NB[3], rdf_total_NN[3],
                 rdf_total_tree[3], rdf_total_support[3],
                 rdf_total_log[3], rdf_total_extra[3], rdf_total_random[3], rdf_time_create[0], rdf_time_train[0],
                 rdf_time_train[1], rdf_time_train[2], rdf_time_train[3], rdf_time_train[4], rdf_time_train[5],
                 rdf_time_train[6],
                 rdf_time_test[0], rdf_time_test[1], rdf_time_test[2], rdf_time_test[3], rdf_time_test[4],
                 rdf_time_test[5], rdf_time_test[6], rdf_memory[0]])

    #
    #     #print(f'AUC LR: {accuracy_score(y_test, y_pred_1)}')
    # if ink_var:
    #     print('---INK----')
    #     print('INK Naive bayes')
    #     print(np.mean(ink_total_NB))
    #     print(np.std(ink_total_NB))
    #     print('INK Neirest neighbors')
    #     print(np.mean(ink_total_NN))
    #     print(np.std(ink_total_NN))
    #     print('INK Decision Tree')
    #     print(np.mean(ink_total_tree))
    #     print(np.std(ink_total_tree))
    #     print('INK SVC')
    #     print(np.mean(ink_total_support))
    #     print(np.std(ink_total_support))
    #     print('INK Logreg')
    #     print(np.mean(ink_total_log))
    #     print(np.std(ink_total_log))
    #     print('---')
    #     print('INK create time')
    #     print(np.mean(ink_time_create))
    #     print(np.std(ink_time_create))
    #     print('INK train time')
    #     print(np.mean(ink_time_train))
    #     print(np.std(ink_time_train))
    #     print('INK test time')
    #     print(np.mean(ink_time_test))
    #     print(np.std(ink_time_test))
    #     print('---')
    #     print('INK embedding size')
    #     print(np.mean(ink_memory))
    #     print(np.std(ink_memory))
    #
    # print('---RDF2Vec----')
    # print('RDF2Vec Naive bayes')
    # print(np.mean(rdf_total_NB))
    # print(np.std(rdf_total_NB))
    # print('RDF2Vec Neirest neighbors')
    # print(np.mean(rdf_total_NN))
    # print(np.std(rdf_total_NN))
    # print('RDF2Vec Decision Tree')
    # print(np.mean(rdf_total_tree))
    # print(np.std(rdf_total_tree))
    # print('RDF2Vec SVC')
    # print(np.mean(rdf_total_support))
    # print(np.std(rdf_total_support))
    # print('RDF2Vec Logreg')
    # print(np.mean(rdf_total_log))
    # print(np.std(rdf_total_log))
    # print('---')
    # print('RDF create time')
    # print(np.mean(rdf_time_create))
    # print(np.std(rdf_time_create))
    # print('RDF train time')
    # print(np.mean(rdf_time_train))
    # print(np.std(rdf_time_train))
    # print('RDF test time')
    # print(np.mean(rdf_time_test))
    # print(np.std(rdf_time_test))
    # print('---')
    # print('RDF embedding size')
    # print(np.mean(rdf_memory))
    # print(np.std(rdf_memory))

