
import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from ink.base.connectors import StardogConnector
from ink.base.structure import InkExtractor
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.feature_selection import VarianceThreshold
###
""" parameters """
if __name__ == "__main__":

    rel = False

    dataset = 'AIFB'#'BGS'
    dir_kb = 'data_node_class/'+dataset
    file = 'AIFB.n3'#'rdf_am-data.ttl'

    format = 'n3'#'ttl'

    #depth = 2

    train = 'data_node_class/'+dataset+'/'+dataset+'_train.tsv'
    test = 'data_node_class/'+dataset+'/'+dataset+'_test.tsv'
    #train = 'mela/train.csv'
    #test = 'mela_tes'

    excludes = ['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']#['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']

    label_name = 'label_affiliation'#'label_cateogory'#'label_lithogenesis'#'label_mutagenic'#'label_affiliation'
    items_name = 'person'#'proxy'#'rock'#'bond'#'person'

    #pos_file = 'mela/pos_mela.txt'
    #neg_file = 'mela/neg_mela.txt'

    df_train = pd.read_csv(train, delimiter='\t')
    df_test = pd.read_csv(test, delimiter='\t')

    data = pd.concat([df_train, df_test])

    le = preprocessing.LabelEncoder()
    df_train['label'] = le.fit_transform(df_train[label_name])
    df_test['label'] = le.transform(df_test[label_name])

    print(df_train['label'].value_counts())
    print(df_test['label'].value_counts())

    pos_file = set(['<' + x + '>' for x in data[items_name].values])


    total_NB = []
    total_NN = []
    total_tree = []
    total_support = []
    total_log = []

    details = {'endpoint': 'http://localhost:5820'}
    connector = StardogConnector(details, dataset)
    connector.upload_kg(dir_kb+'/'+file)

    for _ in tqdm(range(5)):
        extractor = InkExtractor(connector, verbose=True)
        X_train, y_train = extractor.create_dataset(4, pos_file, set(), excludes)
        extracted_data = extractor.fit_transform(X_train, counts=True, levels=True)

        df_data = pd.DataFrame.sparse.from_spmatrix(extracted_data[0])
        df_data.index = [x[1:-1] for x in extracted_data[1]]
        df_data.columns = extracted_data[2]

        # split in test & train

        #print(df_data.shape)
        print(df_data.shape)
        threshold_n=0.75
        sel = VarianceThreshold(threshold=(threshold_n* (1 - threshold_n) ))
        sel_var=sel.fit_transform(df_data)
        df_data = df_data[df_data.columns[sel.get_support(indices=True)]]
        print(df_data.shape)
        #print(len([c for c in list(df_data.columns) if df_data[c].sum() == 1]))
        #df_data = df_data[[c for c in list(df_data.columns) if df_data[c].sum() > 1]]
        #print(df_data.shape)

        df_train_extr = df_data[df_data.index.isin(df_train[items_name].values)]#df_data.loc[[df_train['proxy']],:]
        df_test_extr = df_data[df_data.index.isin(df_test[items_name].values)]#df_data.loc[[df_test['proxy']],:]

        df_train_extr = df_train_extr.merge(df_train[[items_name,'label']], left_index=True, right_on=items_name)
        df_test_extr = df_test_extr.merge(df_test[[items_name, 'label']], left_index=True, right_on=items_name)


        ####
        X = df_train_extr.drop(['label',items_name], axis=1).values
        y = df_train_extr['label'].values

        #for train_index, test_index in skf.split(X, y):
        clf_1 = KNeighborsClassifier(n_neighbors=3)
        clf_2 = MultinomialNB(alpha=0)
        clf_3 = DecisionTreeClassifier()
        clf_4 = GridSearchCV(SVC(), {'C':[10**-3, 10**-2, 0.1, 1, 10, 10**2, 10**3]}, cv=3, n_jobs=4)
        clf_5 = GridSearchCV(LogisticRegression(), {'C':[10**-3, 10**-2, 0.1, 1, 10, 10**2, 10**3], 'max_iter':[10000]}, cv=3, n_jobs=4)

        clf_1.fit(X,y)
        clf_2.fit(X,y)
        clf_3.fit(X,y)
        clf_4.fit(X,y)
        clf_5.fit(X,y)

        print('fit done')

        y_pred_1 = clf_1.predict(df_test_extr.drop(['label',items_name], axis=1).values)
        y_pred_2 = clf_2.predict(df_test_extr.drop(['label',items_name], axis=1).values)
        y_pred_3 = clf_3.predict(df_test_extr.drop(['label',items_name], axis=1).values)
        y_pred_4 = clf_4.predict(df_test_extr.drop(['label',items_name], axis=1).values)
        y_pred_5 = clf_5.predict(df_test_extr.drop(['label',items_name], axis=1).values)

        total_NN.append(accuracy_score(df_test_extr['label'].values, y_pred_1))
        total_NB.append(accuracy_score(df_test_extr['label'].values, y_pred_2))
        total_tree.append(accuracy_score(df_test_extr['label'].values, y_pred_3))
        total_support.append(accuracy_score(df_test_extr['label'].values, y_pred_4))
        total_log.append(accuracy_score(df_test_extr['label'].values, y_pred_5))


        #print(f'AUC LR: {accuracy_score(y_test, y_pred_1)}')

    print('Naive bayes')
    print(np.mean(total_NB))
    print(np.std(total_NB))
    print('Neirest neighbors')
    print(np.mean(total_NN))
    print(np.std(total_NN))
    print('Decision Tree')
    print(np.mean(total_tree))
    print(np.std(total_tree))
    print('SVC')
    print(np.mean(total_support))
    print(np.std(total_support))
    print('Logreg')
    print(np.mean(total_log))
    print(np.std(total_log))
