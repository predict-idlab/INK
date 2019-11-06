from graph import KnowledgeGraph, remove_namespace
import pandas as pd
from sklearn.model_selection import train_test_split
import ruleset as rs
import rdflib
import numpy as np
from sklearn import preprocessing



kg = KnowledgeGraph("family-benchmark.owl", "application/rdf+xml")
qres = kg.query("""SELECT ?s WHERE {
                    ?s a <http://www.benchmark.org/family#Person>
               }""")

df = pd.DataFrame()
for row in qres:
    kg.create_neighbours(row[0].n3(), 3, ["http://www.benchmark.org/family#Son"])

series = []
ind = []
for k in kg.ind_instances:
    ind.append(k)
    data = kg.ind_instances[k]
    series.append(data)

df = pd.DataFrame(series)
df.index = ind

import math
def isnan(x):
    if isinstance(x, (int, float, complex)) and math.isnan(x):
        return True

df = df.apply(lambda x:x.apply(lambda x:set() if isnan(x) else x))
#print(df)

train_file = "Family_train.tsv"
test_file = "Family_test.tsv"
ind_col = "person"
lbl = "label"

gt = pd.read_csv(train_file, sep='\t', index_col=ind_col)[lbl]
gt.index = gt.index.map(remove_namespace)
gt = gt.apply(remove_namespace)
print(df)
X_train = pd.merge(df, gt, left_index=True, right_index=True, how='inner')

le = preprocessing.LabelEncoder()
X_train[lbl] = le.fit_transform(X_train[lbl].values)
X_train[lbl] = X_train[lbl].apply(lambda x: 1 if x==1 else 0)
y_train = X_train[lbl].values
X_train.drop(lbl, axis=1, inplace=True)

# gt = pd.read_csv(test_file, sep='\t', index_col=ind_col)[lbl]
# gt.index = gt.index.map(remove_namespace)
# gt = gt.apply(remove_namespace)

# X_test = pd.merge(df, gt, left_index=True, right_index=True, how='inner')
#
# X_test[lbl] = le.fit_transform(X_test[lbl].values)
# X_test[lbl] = X_test[lbl].apply(lambda x: 1 if x==1 else 0 )
# y_test = X_test[lbl].values
# X_test.drop(lbl, axis=1, inplace=True)

model = rs.BayesianRuleSet(method='forest')
model.fit(X_train, y_train)
# yhat = model.predict(X_test)
# TP, FP, TN, FN = rs.get_confusion(yhat, y_test)
# print(TP, FP, TN, FN)
# print((TP+TN+0.0)/len(yhat))