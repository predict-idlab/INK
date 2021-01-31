
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
        extractor = InkExtractor(connector, verbose=True)
        X_train, _ = extractor.create_dataset(depth, pos_file, set(), excludes, jobs=4)
        extracted_data = extractor.fit_transform(X_train, counts=False, levels=False, float_rpr=True)

        df_data = pd.DataFrame.sparse.from_spmatrix(extracted_data[0])
        df_data.index = [x[1:-1] for x in extracted_data[1]]
        df_data.columns = extracted_data[2]


        df_train_extr = df_data[df_data.index.isin(df_train[items_name].values)]  # df_data.loc[[df_train['proxy']],:]
        df_test_extr = df_data[df_data.index.isin(df_test[items_name].values)]  # df_data.loc[[df_test['proxy']],:]

        df_train_extr = df_train_extr.merge(df_train[[items_name, 'label']], left_index=True, right_on=items_name)
        df_test_extr = df_test_extr.merge(df_test[[items_name, 'label']], left_index=True, right_on=items_name)

        ####
        X = df_train_extr.drop(['label', items_name], axis=1).values
        y = df_train_extr['label'].values

        clf_3 = DecisionTreeClassifier()


        clf_3.fit(X, y)

        from sklearn import tree

        y_pred_3 = clf_3.predict(df_test_extr.drop(['label', items_name], axis=1).values)

        tree.export_graphviz(clf_3,feature_names = df_data.columns,class_names=['FLUV', 'GLACI'], out_file="tree.dot")