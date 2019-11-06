############################
# Bayesian Rule Set mining
# By Tong Wang and Peter (Zhen) Li
# 
# reference:
# Wang, Tong, et al. "Bayesian rule sets for interpretable classification." 
# Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016.

from __future__ import print_function
from __future__ import division

import math
import itertools
from itertools import chain, combinations
from bisect import bisect_left
from random import sample
import time
import operator
from collections import Counter, defaultdict
import multiprocessing as mp
from multiprocessing import Pool

import pandas as pd
#from fim import fpgrowth, fim
import numpy as np
from numpy.random import random
from scipy.stats.distributions import poisson, gamma, beta, bernoulli, binom
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class BayesianRuleSet(object):
    '''Find the rule set from the data
    The input data should follow the following format:
    X needs to be a pandas DataFrame
    all the column names can not contain '§' or '<'
    and the column names can not be pure numbers
    The categorical data should be represented in string
    (For example, gender needs to be 'male' or 'female',
     or '0' or '1' to represent male and female respectively.)
    The parser will only recognize this format of data.
    So transform the data set first before using the
    functions.

    The program is very picky on the input data format
    X needs to be a pandas DataFrame,
    y needs to be a nd.array as integer or boolean

    Note
    ----------
    (Sometimes, there will be open range. This is because the 
    cutoffs in the range is empty.)
    
    Parameters
    ----------
    max_rules : int, default 5000
        Maximum number of rules when generating rules

    max_iter : int, default 200
        Maximun number of iteratations to find the rule set

    chians : int, default 1
        Number of chains that run in parallel

    support : int, default 5
        The support is the percentile threshold for the itemset 
        to be selected.

    maxlen : int, default 3
        The maximum number of items in a rule

    #note need to replace all the alpha_1 to alpha_+
    alpha_1 : float, default 100
        alpha_+

    beta_1 : float, default 1
        beta_+

    alpha_2 : float, default 100
        alpha_-

    beta_2 : float, default 1
        beta_-

    alpha_l : float array, shape (maxlen+1,)
        default all elements to be 1

    beta_l : float array, shape (maxlen+1,)
        default corresponding patternSpace

    level : int, default 4
        Number of intervals to deal with numerical continous features
        
    neg : boolean, default True
        Negate the features

    add_rules : list, default empty
        User defined rules to add
        it needs user to add numerical version of the rules

    criteria : str, default 'precision'
        When there are rules more than max_rules,
        the criteria used to filter rules

    greedy_initilization : boolean, default False
        Wether start the rule set using a greedy 
        initilization (according to accuracy)

    greedy_threshold : float, default 0.05
        Threshold for the greedy algorithm
        to find the starting rule set

    propose_threshold : float, default 0.1
        Threshold for a proposal to be accepted

    method : str, default 'forest'
        The method used to generate rules.
        Can be 'fpgrowth' or 'forest'
        Notice that if there are potentially many rules
        then fpgrowth is not a good method as it will
        have memory issue (because the rule screening is
        after rule generations).
        If 'fpgrowth' is used, there has to have 
        fpgrowth preinstalled.
        http://www.borgelt.net/pyfim.html
    rule_adjust : boolean, default True
        If the numerical rules need to be adjusted
        according to the cutoffs.
    binary_input : boolean, default False
        If the input has been binarized, then
        set it to True.
    '''

    def __init__(self, max_rules=5000, max_iter=200, chains=1,
                 support=5, maxlen=3, alpha_1=100, beta_1=1,
                 alpha_2=100, beta_2=1, alpha_l=None, 
                 beta_l=None, level=4, neg=True, 
                 add_rules=[], criteria='precision',
                 greedy_initilization=False, greedy_threshold=0.05,
                 propose_threshold = 0.1,
                 method='forest', forest_size = 500,
                 rule_adjust=True, binary_input=False):
        self.max_rules = max_rules
        self.max_iter = max_iter
        self.chains = chains
        self.support = support
        self.maxlen = maxlen
        self.alpha_1 = alpha_1
        self.beta_1 = beta_1
        self.alpha_2 = alpha_2
        self.beta_2 = beta_2
        self.alpha_l = alpha_l
        self.beta_l = beta_l
        self.level = level
        self.neg = neg
        self.add_rules = add_rules
        self.criteria = criteria
        self.greedy_initilization = greedy_initilization
        self.greedy_threshold = greedy_threshold
        self.propose_threshold = propose_threshold
        self.method = method
        self.forest_size = forest_size
        self.rule_adjust = rule_adjust
        self.binary_input = binary_input
        self.predicted_rules = []

    def transform(self, df, neg=True):
        '''Transform input dataframe to binary data
        Negation of the data will be done at this stage
        Parameters
        ----------
        df : pandas dataframe
             data
        
        neg : boolean, default True
              count the negation or not

        Note
        ----------
        Attribute names should not contain '§' or '<'
        because these two characters are used to distinguish
        categorical and numerical values
        '''
    
        self.attributeNames = df.columns
        # the data type can not be 0 or 1 for catagorical...
        level = self.level
        # df_cat is categorical part of df
        df_cat = df.loc[:, df.dtypes==object]
        # df_num is numerical part of df
        df_num = df.loc[:, (df.dtypes=='float64')^(df.dtypes=='int64')]
        df_cat_columns = df_cat.columns
        for col in df_cat_columns:
            if type(df_cat[col].values[0]) is set:
                items = np.unique([y for x in df_cat[col].values for y in x])#np.unique(df_cat[col])
                # if there are two categories, there is no need to negate
                if len(items) == 2:
                    for item in items:
                        df_cat[col+'§'+str(item)] = [True if item in x else False for x in df_cat[col].values]
                else:
                    for item in items:
                        df_cat.loc[:, col+'§'+str(item)] = [True if item in x else False for x in df_cat[col].values]
                        if neg:
                            df_cat.loc[:, col+'§'+str(item)+'§neg'] = [True if item not in x else False for x in df_cat[col].values]
                df_cat.drop(col, axis=1, inplace=True)
            else:
                items = np.unique(df_cat[col])
                if len(items) == 2:
                    for item in items:
                        df_cat[col+'§'+str(item)] = [True if item == x else False for x in df_cat[col].values]
                else:
                    for item in items:
                        df_cat.loc[:, col+'§'+str(item)] = [True if item == x else False for x in df_cat[col].values]
                        if neg:
                            df_cat.loc[:, col+'§'+str(item)+'§neg'] = [True if item != x else False for x in df_cat[col].values]
                df_cat.drop(col, axis=1, inplace=True)
        df_num_columns = df_num.columns
        for col in df_num_columns:
            for q in range(level):
                l = df_num[col].quantile(float(q)/level)
                u = df_num[col].quantile(float(q+1)/level)
                # the negation of this part should always exits
                # because in this case, we can have intervals
                if q == 0:
                    df_num.loc[:, col+'<'+str(u)] = df_num[col]<u
                elif q == level-1:
                    df_num.loc[:, str(l)+'<'+col] = df_num[col]>=l
                else:
                    df_num.loc[:, str(l)+'<'+col] = df_num[col]>=l
                    df_num.loc[:, col+'<'+str(u)] = df_num[col]<u
            df_num.drop(col, axis=1, inplace=True)
        return pd.concat([df_cat, df_num], axis=1)


    def set_parameters(self, X):
        # number of possible rules, i.e. rule space italic(A) prior
        self.patternSpace = np.ones(self.maxlen+1)
        # This patternSpace is an approximation
        # because the original code allows
        # the following situation, take tic-tac-toe
        # 1_O == 1 and 1_O_neg == 1, which is impossible
        numAttributes = len(X.columns)
        for i in range(1, self.maxlen+1):
            tmp = 1
            for j in range(numAttributes-i+1,numAttributes+1): 
                tmp *= j
            self.patternSpace[i] = tmp / math.factorial(i)
		
        if self.alpha_l == None:
            self.alpha_l = [1 for i in range(self.maxlen+1)]
        if self.beta_l == None:
            self.beta_l = [(self.patternSpace[i]*100+1) for i in range(self.maxlen+1)]
            
    def precompute(self, y):
        TP, FP, TN, FN = sum(y), 0, len(y)-sum(y), 0
        # self.Lup : p(S|A;alpha_+,beta_+,alpha_-,beta_-)
        # conference paper formula(6)
        self.Lup = (log_betabin(TP, TP+FP, self.alpha_1, self.beta_1)
                    + log_betabin(TN, FN+TN, self.alpha_2, self.beta_2))
        # self.const_denominator : log((|Al|+beta_l-1)/(alpha_l+|Al|-1))
        # conference paper formula(9) denominator
        self.const_denominator = [np.log((self.patternSpace[i] + self.beta_l[i] - 1)
                                         / (self.patternSpace[i] + self.alpha_l[i] - 1))
                                         for i in range(self.maxlen+1)]
        Kn_count = np.zeros(self.maxlen+1, dtype=int)
        # P0 : maximum prior
        # Ml=0, |Al|= rule space
        # conference paper formula(3)
        # because of log property, + is *
        self.P0 = sum([log_betabin(Kn_count[i], self.patternSpace[i], self.alpha_l[i],
                                   self.beta_l[i]) for i in range(1, self.maxlen+1)])

    def compute_cutoffs(self, X, y):
        # cutoffs have the format:
        # key : name
        # value : (cutoffs, interval)
        cutoffs = dict()
        for col in X.columns:
            if (X[col].dtype == 'int64') | (X[col].dtype == 'float64'):
                tmps = sorted(list(zip(X[col].values, y)))
                cutoff = []
                interval = (X[col].max() - X[col].min()) / self.level
                for i, tmp in enumerate(tmps):
                    try:
                        if (tmp[1] == 1) & (tmps[i+1][1] == 0):
                            if tmp[0] not in cutoff:
                                cutoff.append(tmp[0])
                        elif (tmp[1] == 0) & (tmps[i+1][1] == 1):
                            if tmps[i+1][0] not in cutoff:
                                cutoff.append(tmps[i+1][0])
                        else:
                            pass
                    except:
                        pass
                cutoffs[col] = (cutoff, interval)
        self.cutoffs = cutoffs

    def generate_rules(self, X_trans, y):
 
        itemNames = dict()
        for i, item in enumerate(X_trans.columns):
            itemNames[i+1] = item
        self.itemNames = itemNames

        if self.method == 'fpgrowth':
            from fim import fpgrowth, fim
            items = np.arange(1, len(X_trans.columns)+1)
            itemMatrix = (X_trans * items).as_matrix()
            itemMatrix_numerical = np.array([row[row>0] for row in itemMatrix])
            rules = fpgrowth(itemMatrix_numerical[np.where(y==1)].tolist(), 
                             supp=self.support, zmin=1,
                             zmax=self.maxlen)
            self.rules = [sorted(rule[0]) for rule in rules]
        else:
            items = np.arange(1, len(X_trans.columns)+1)
            rules = []
            for length in range(1, self.maxlen+1):
                # if the data is too small, it will complaine
                # the n_estimators can not larger than the 
                # possible trees
                n_estimators = self.forest_size * length
                clf = RandomForestClassifier(n_estimators=n_estimators,
                                             max_depth=length)
                clf.fit(X_trans, y)
                for n in range(n_estimators):
                    rules.extend(extract_rules(clf.estimators_[n], items))
            # To-do: not sure which one is faster, needs to test on a large dataset
            rules = [list(x) for x in set(tuple(np.sort(y)) for y in rules)]
            # rules = [list(x) for x in remove_duplicates(tuple(np.sort(y)) for y in rules)]
            self.rules = rules
        # this needs to be modified, because
        # it needs user to add numerical version of the rules
        for add_rule in self.add_rules:
            if np.sort(add_rule).tolist() not in self.rules:
                self.rules.append(add_rule)

    def screen_rules(self, X_trans, y):
        tmp_rules_len = [len(rule) for rule in self.rules]
        ruleMatrix = np.zeros((len(self.rules), len(X_trans.columns)), dtype=int)
        for i, rule in enumerate(self.rules):
            for j in rule:
                ruleMatrix[i][j-1] = 1
        ruleMatrix = sparse.csc_matrix(ruleMatrix.transpose())

        mat = (sparse.csc_matrix(X_trans) * ruleMatrix).todense()
        # Z is the matrix for data points covered by rules
        Z = (mat==tmp_rules_len)
        Zpos = Z[np.where(y>0)]
        # TP for each rule
        TP = np.asarray(np.sum(Zpos, axis=0))[0]
        # supp is shreshold percentile of how TP a rule is
        supp_select = np.where(TP>=self.support*sum(y)/100)[0]
        
        if len(supp_select)<=self.max_rules:
            self.rules = np.asarray(self.rules)[supp_select]
            RMatrix = np.array(Z[:, supp_select])

            self.rules_len = [len(rule) for rule in self.rules]
            self.supp = np.asarray(np.sum(Z, axis=0))[0][supp_select]
        else:
            FP = np.array(np.sum(Z, axis=0))[0] - TP
            TN = len(y) - np.sum(y) - FP
            FN = np.sum(y) - TP
            p1 = TP.astype(float) / (TP+FP)
            p2 = FN.astype(float) / (FN+TN)
            pp = (TP+FP).astype(float) / (TP+FP+TN+FN)

            if self.criteria == 'precision':
                select = np.argsort(p1[supp_select])[::-1][:self.max_rules].tolist()
            elif self.criteria == 'specificity':
                p3 = TN.astype(float) / (TN+FP)
                select = np.argsort(p3[supp_select])[::-1][:self.max_rules].tolist()
            elif self.criteria == 'sensitivity':
                p4 = TP.astype(float) / (TP+FN)
                select = np.argsort(p4[supp_select])[::-1][:self.max_rules].tolist()
            else:
                cond_entropy = (-pp*(p1*np.log(p1) + (1-p1)*np.log(1-p1))
                                - (1-pp)*(p2*np.log(p2)
                                + (1-p2)*np.log(1-p2)))
                cond_entropy[p1*(1-p1)==0] = (-((1-pp)*(p2*np.log(p2)
                                              + (1-p2)*np.log(1-p2)))[p1*(1-p1)==0])
                cond_entropy[p2*(1-p2)==0] = (-(pp*(p1*np.log(p1)
                                              + (1-p1)*np.log(1-p1)))[p2*(1-p2)==0])
                cond_entropy[p1*(1-p1)*p2*(1-p2)==0] = 0
                pos = (TP+FN).astype(float)/(TP+FP+TN+FN)
                info = - pos * np.log(pos) - (1-pos)*np.log(1-pos)
                info[np.where((pos==1)| (pos==0))[0]] = 0
                IGR = (info - cond_entropy)/info
                IGR[np.where(info==0)[0]] = 0
                select = np.argsort(IGR[supp_select])[::-1][:self.max_rules].tolist()
            ind = list(supp_select[select])
            self.rules = [self.rules[i] for i in ind]
            RMatrix = np.array(Z[:,ind])
            self.rules_len = [len(rule) for rule in self.rules]
            self.supp = np.asarray(np.sum(Z,axis=0))[0][ind]
        return RMatrix        
    
    
    def screen_rules_interval(self, X_trans, y, interval_size=50000):
        num_iterations = int(np.ceil(len(self.rules)/interval_size))
        RMatrix_total = np.zeros((X_trans.shape[0], 1), dtype=int)
        rules_self = []
        for num_iteration in range(num_iterations):
            tmp_rules_len = [len(rule) for rule in self.rules[num_iteration*interval_size:
                num_iteration*interval_size+interval_size]]
            ruleMatrix = np.zeros((interval_size, len(X_trans.columns)), dtype=int)
            
            for i, rule in enumerate(self.rules[num_iteration*interval_size:
                num_iteration*interval_size+interval_size]):
                for j in rule:
                    ruleMatrix[i][j-1] = 1
            ruleMatrix = sparse.csc_matrix(ruleMatrix.transpose())

            mat = (sparse.csc_matrix(X_trans) * ruleMatrix).todense()
            Z = (mat==tmp_rules_len)
            Zpos = Z[np.where(y>0)]

            TP = np.asarray(np.sum(Zpos, axis=0))[0]
            supp_select = np.where(TP>=self.support*sum(y)/100)[0]
        
            if len(supp_select)<=self.max_rules:
                #self.rules = np.asarray(self.rules)[supp_select]
                rules_self.extend(np.asarray(self.rules)[supp_select])
                RMatrix = np.array(Z[:, supp_select])
            else:
                FP = np.array(np.sum(Z, axis=0))[0] - TP
                TN = len(y) - np.sum(y) - FP
                FN = np.sum(y) - TP
                p1 = TP.astype(float) / (TP+FP)
                p2 = FN.astype(float) / (FN+TN)
                pp = (TP+FP).astype(float) / (TP+FP+TN+FN)

                if self.criteria == 'precision':
                    select = np.argsort(p1[supp_select])[::-1][:self.max_rules].tolist()
                elif self.criteria == 'specificity':
                    p3 = TN.astype(float) / (TN+FP)
                    select = np.argsort(p3[supp_select])[::-1][:self.max_rules].tolist()
                elif self.criteria == 'sensitivity':
                    p4 = TP.astype(float) / (TP+FN)
                    select = np.argsort(p4[supp_select])[::-1][:self.max_rules].tolist()
                else:
                    cond_entropy = (-pp*(p1*np.log(p1) + (1-p1)*np.log(1-p1))
                                    - (1-pp)*(p2*np.log(p2)
                                    + (1-p2)*np.log(1-p2)))
                    cond_entropy[p1*(1-p1)==0] = (-((1-pp)*(p2*np.log(p2)
                                                  + (1-p2)*np.log(1-p2)))[p1*(1-p1)==0])
                    cond_entropy[p2*(1-p2)==0] = (-(pp*(p1*np.log(p1)
                                                  + (1-p1)*np.log(1-p1)))[p2*(1-p2)==0])
                    cond_entropy[p1*(1-p1)*p2*(1-p2)==0] = 0
                    pos = (TP+FN).astype(float)/(TP+FP+TN+FN)
                    info = - pos * np.log(pos) - (1-pos)*np.log(1-pos)
                    info[np.where((pos==1)| (pos==0))[0]] = 0
                    IGR = (info - cond_entropy)/info
                    IGR[np.where(info==0)[0]] = 0
                    select = np.argsort(IGR[supp_select])[::-1][:self.max_rules].tolist()
                ind = list(supp_select[select])
                rules_self.extend([self.rules[i] for i in ind])
                #self.rules = [self.rules[i] for i in ind]
                RMatrix = np.array(Z[:,ind])
            RMatrix_total = np.concatenate((RMatrix_total, RMatrix), axis=1)
        RMatrix_total = RMatrix_total[:,1:]
        self.rules = rules_self    
        self.rules_len = [len(rule) for rule in self.rules]
        self.supp = np.asarray(np.sum(RMatrix_total,axis=0))[0]
        return RMatrix_total
    
    def modify_rule(self, X, X_trans, y, rule_idx):
        # first check out the categorical rules and 
        # numerical rules
        rule = self.rules[rule_idx]
        num_items = []
        cat_items = []
        for item in rule:
            if '§' in self.itemNames[item]:
                cat_items.append(item)
            if '<' in self.itemNames[item]:
                num_items.append(item)

        # if there is no numerical rules to modify
        # then return
        if len(num_items) == 0:
            # None means the rule has its original
            # meaning
            # if self.rule_explainations is None
            # then could use RMatrix
            #self.rule_explainations[rule_idx] = None
            return rule_idx
        working_idx = []
        if len(cat_items) == 0:
            working_idx = np.arange(X_trans.shape[0])
            rule_Z = np.ones(X_trans.shape[0], dtype=int)
        else:
            tmp_cat_items_columns = X_trans[self.itemNames[cat_items[0]]]
            if len(cat_items) > 1:
                for cat_item_ith in range(1, len(cat_items)):
                    tmp_cat_items_columns = (tmp_cat_items_columns 
                                             & X_trans[self.itemNames[cat_items[cat_item_ith]]])
            rule_Z = tmp_cat_items_columns
            working_idx = np.where(rule_Z)[0]
        
        if len(working_idx) == 0:
            #self.rule_explainations[rule_idx] = None
            return rule_idx
      

        # feature < value
        # means min < feature < value
        # value < feature 
        # means value < feature < max
        temp_itemNames = dict()
        num_items_ext = []
        for num_item in num_items:
            name_meaning = self.itemNames[num_item].split('<')
            if name_meaning[0] in self.attributeNames:
                temp_itemName = str(X[name_meaning[0]].min())+'<'+name_meaning[0]
            else:
                temp_itemName = name_meaning[1]+'<'+str(X[name_meaning[1]].max())
            if temp_itemName not in temp_itemNames:
                temp_itemNames[temp_itemName] = 1
            num_items_ext.append(temp_itemName)

        num_items.extend(num_items_ext)

        # build the matrix for numerical rules
        # To-do :  can make this part smarter by
        #          using index properly

        working_matrix = np.zeros((len(working_idx), 1), dtype=int)
        working_y = y[working_idx]
        # 0 means feature < number
        # 1 means number < feature
        # (item_number for self.itemNames,
        #  name_meaning
        #  feature_meaning,
        #  cutoff, # this might not be important 
        #  ith_item)
        working_item_list = []
        # (ith_item_length)
        working_item_list_length = []
        valid_working_items = -1
        for num_item in num_items:
            if num_item in temp_itemNames:
                name_meaning = num_item.split('<')
            else:
                name_meaning = self.itemNames[num_item].split('<')
            # may write a function here
            if name_meaning[0] in self.attributeNames:
                working_cutoffs = np.array(self.cutoffs[name_meaning[0]][0])
                cutoff_min = float(name_meaning[1]) - self.cutoffs[name_meaning[0]][1]
                cutoff_max = float(name_meaning[1]) + self.cutoffs[name_meaning[0]][1]
                working_cutoffs = working_cutoffs[working_cutoffs>=cutoff_min]
                working_cutoffs = working_cutoffs[working_cutoffs<=cutoff_max]
                if len(working_cutoffs) == 0:
                    continue
                else:
                    valid_working_items += 1
                    working_X = X[name_meaning[0]].values[working_idx]
                    for working_cutoff in working_cutoffs:
                        working_matrix = np.concatenate((working_matrix,
                                                         (working_X<working_cutoff).reshape(-1,1)),
                                                         axis=1)
                        working_item_list.append((num_item, name_meaning[0], 0, 
                                                  working_cutoff, valid_working_items))
                    working_item_list_length.append(len(working_cutoffs))
            else:
                working_cutoffs = np.array(self.cutoffs[name_meaning[1]][0])
                cutoff_min = float(name_meaning[0]) - self.cutoffs[name_meaning[1]][1]
                cutoff_max = float(name_meaning[0]) + self.cutoffs[name_meaning[1]][1]
                working_cutoffs = working_cutoffs[working_cutoffs>=cutoff_min]
                working_cutoffs = working_cutoffs[working_cutoffs<=cutoff_max]
                if len(working_cutoffs) == 0:
                    continue
                else:
                    valid_working_items += 1
                    working_X = X[name_meaning[1]].values[working_idx]
                    for working_cutoff in working_cutoffs:
                        working_matrix = np.concatenate((working_matrix,
                                                         (working_X>=working_cutoff).reshape(-1,1)),
                                                         axis=1)
                        working_item_list.append((num_item, name_meaning[1], 1, 
                                                  working_cutoff, valid_working_items))
                    working_item_list_length.append(len(working_cutoffs))
        working_matrix = working_matrix[:,1:]
        if valid_working_items == -1:
            #self.rule_explainations[rule_idx] = None
            return rule_idx
            # working feature should be a list
        
        rule_meaning = []
        for ith_item in range(valid_working_items+1):
            working_features = np.arange(working_matrix.shape[1])
            if working_matrix.shape[1] != 1:
                decisionTree = DecisionTreeClassifier(max_depth=1)
                decisionTree.fit(working_matrix, working_y)
                working_feature = extract_rules(decisionTree, working_features)[0][0]
            else:
                working_feature = 0
            working_feature_name = working_item_list[working_feature][1]
            working_feature_meaning = working_item_list[working_feature][2]
            # use accuracy to find the best cutoffs
            feature_accuracy = []
            feature_interval = find_interval(working_feature, working_item_list_length)
            starting_idx = sum(working_item_list_length[:feature_interval])
            ending_idx = sum(working_item_list_length[:feature_interval+1])
            cutoff_list = list(range(starting_idx, ending_idx))
            for ith_cutoff in cutoff_list:
                TP, FP, TN, FN = get_confusion(working_matrix[:,ith_cutoff], working_y)
                feature_accuracy.append((TP+TN)/(TP+TN+FP+FN))
            best_item = working_item_list[cutoff_list[np.argsort(feature_accuracy)[::-1][0]]]
            update_idx = np.where(working_matrix[:,cutoff_list[np.argsort(feature_accuracy)[::-1][0]]])
            if working_feature_meaning == 0:
                rule_meaning.append(working_feature_name+'<'
                                    +str(best_item[3]))
                rule_Z = rule_Z & (X[working_feature_name]<best_item[3])
            else:
                rule_meaning.append(str(best_item[3])+'<'
                                    +working_feature_name)
                rule_Z = rule_Z & (X[working_feature_name]>=best_item[3])

            # now update working_matrix
            # working_item_list and working_item_list_length
            working_matrix = working_matrix[update_idx]
            working_y = working_y[update_idx[0]]
            del(working_item_list_length[feature_interval])
            working_item_list = working_item_list[:starting_idx] + working_item_list[ending_idx:]
            update_shape = working_matrix.shape[0]
            if update_shape == 0:
                break
            working_matrix = np.concatenate((working_matrix[:,:starting_idx].reshape(update_shape, -1), 
                                             working_matrix[:,ending_idx:].reshape(update_shape, -1)), 
                                             axis=1)
        for cat_item in cat_items:
            rule_meaning.append(self.itemNames[cat_item])
        self.rule_explainations[rule_idx] = (rule_meaning, rule_Z)

    
    def greedy_init(self, X, X_trans, y, RMatrix):
        greedy_rules = []
        stop_condition = max(int(RMatrix.shape[0]*self.greedy_threshold), 100)

        idx = np.arange(RMatrix.shape[0])
        while True:
            TP = np.sum(RMatrix[idx], axis=0)
            rule = sample(np.where(TP==TP.max())[0].tolist(), 1)
            self.modify_rule(X, X_trans, y, rule[0])
            greedy_rules.extend(rule)
            Z = self.find_rules_Z(RMatrix, greedy_rules)
            #idx = np.where(RMatrix[:, rule]==False)
            idx = np.where(Z==False)
            if np.sum(RMatrix[idx], axis=0).max() < stop_condition:
                return greedy_rules

    def normalize(self, rules_new):
        try:
            rules_len = [len(self.rules[index]) for index in rules_new]
            rules = [rules_new[i] for i in np.argsort(rules_len)[::-1][:len(rules_len)]]
            p1 = 0
            while p1<len(rules):
                for p2 in range(p1+1,len(rules),1):
                    if set(self.rules[rules[p2]]).issubset(set(self.rules[rules[p1]])):
                        rules.remove(rules[p1])
                        p1 -= 1
                        break
                p1 += 1
            return rules[:]
        except:
            return rules_new[:]

    def find_rules_Z(self, RMatrix, rules):
        if len(rules) == 0:
            return np.zeros(RMatrix.shape[0], dtype=int)
        Z = np.zeros(RMatrix.shape[0], dtype=int)
        for rule in rules:
            if self.rule_explainations.get(rule) == None:
                Z = RMatrix[:, rule] + Z
            else:
                Z = self.rule_explainations[rule][1] + Z
        Z = Z > 0
        return Z


    def propose(self, rules_curr, rules_norm, 
                nRules, X, X_trans, y, RMatrix):
        yhat = self.find_rules_Z(RMatrix, rules_curr)
        incorr = np.where(y!=yhat)[0]
        rules_curr_len = len(rules_curr)

        if len(incorr) == 0:
            clean = True
            move = ['clean']
        else:
            clean = False
            ex = sample(list(incorr), 1)[0]
            t = random()
            if y[ex] == 1 or rules_curr_len == 1:
                if t < 1.0/2 or rules_curr_len == 1:
                    move = ['add']
                else:
                    move = ['cut', 'add']
            else:
                if t < 1.0/2:
                    move = ['cut']
                else:
                    move = ['cut', 'add']
        # 'cut' a rule
        if move[0] == 'cut':
            try:
                if random() < self.propose_threshold:
                    candidate = []
                    for rule in rules_curr:
                        if self.rule_explainations.get(rule) == None:
                            if RMatrix[ex, rule]:
                                candidate.append(rule)
                        else:
                            if self.rule_explainations[rule][1][ex]:
                                candidate.append(rule)
                    if len(candidate) == 0:
                        candidate = rules_curr
                    cut_rule = sample(candidate, 1)[0]
                else:
                    p = []
                    all_sum = np.zeros(RMatrix.shape[0], dtype=int)
                    for rule in rules_curr:
                        if self.rule_explainations.get(rule) == None:
                            all_sum = all_sum + RMatrix[:, rule]
                        else:
                            all_sum = all_sum + self.rule_explainations[rule][1].astype(int)
            
                    for ith_rule, rule in enumerate(rules_curr):
                        if self.rule_explainations.get(rule) == None:
                            yhat = (all_sum - RMatrix[:, rule]) > 0
                        else:
                            yhat = (all_sum - self.rule_explainations[rule][1].astype(int)) > 0
                        TP, FP, TN, FN  = get_confusion(yhat, y)
                        p.append(TP.astype(float)/(TP+FP+1))
                    p = [x - min(p) for x in p]
                    p = np.exp(p)
                    p = np.insert(p,0,0)
                    p = np.array(list(accumulate(p)))
                    if p[-1]==0:
                        index = sample(list(range(len(rules_curr))),1)[0]
                    else:
                        p = p/p[-1]
                        index = find_lt(p,random())
                    cut_rule = rules_curr[index]
                rules_curr.remove(cut_rule)
                rules_norm = self.normalize(rules_curr)
                move.remove('cut')
            except:
                move.remove('cut')

        # 'add' a rule        
        if len(move) > 0 and move[0] == 'add':
            if y[ex] == 1:
                select = np.where((self.supp>self.C[-1]) & RMatrix[ex] > 0)[0]
            else:
                select = np.where((self.supp>self.C[-1]) & ~RMatrix[ex] > 0)[0]
            if len(select) > 0:
                if random() < self.propose_threshold:
                    add_rule = sample(select.tolist(),1)[0]
                else:
                    Yhat_neg_index = np.where(~self.find_rules_Z(RMatrix, rules_curr))[0]
                    # In case Yhat_neg_index is []
                    if Yhat_neg_index.shape[0] == 0:
                        return rules_curr, rules_norm
                    mat = RMatrix[Yhat_neg_index.reshape(-1,1), select].transpose() & y[Yhat_neg_index].astype(int)
                    TP = np.sum(mat,axis = 1)
                    FP = np.array(np.sum(RMatrix[Yhat_neg_index.reshape(-1,1), select],axis = 0) - TP)
                    p = (TP.astype(float)/(TP+FP+1))
                    add_rule = select[sample(list(np.where(p==max(p))[0]),1)[0]]
                try:
                    if add_rule not in rules_curr:
                        rules_curr.append(add_rule)
                        if self.rule_adjust and ~self.binary_input:
                            self.modify_rule(X, X_trans, y, add_rule)
                except:
                    1

        if len(move) > 0 and move[0] == 'clean':
            remove = []
            for i, rule in enumerate(rules_norm):
                yhat = np.zeros(RMatrix.shape[0], dtype=int)
                for j, rule_j in enumerate(rules_norm):
                    if (j != i and j not in remove):
                        if self.rule_explainations.get(rule_j) == None:
                            yhat = yhat + RMatrix[:, rule_j]
                        else:
                            yhat = yhat + self.rule_explainations[rule_j][1].astype(int)
                yhat = yhat > 0
                TP, FP, TN, FN = get_confusion(yhat, y)
                if TP + FP==0:
                    remove.append(i)
            for x in remove:
                rules_norm.remove(x)
            return rules_curr, rules_norm
        return rules_curr, rules_norm

    def compute_prob(self, RMatrix, y, rules):
        yhat = self.find_rules_Z(RMatrix, rules)
        self.yhat = yhat
        TP, FP, TN, FN = get_confusion(yhat, y)
        Kn_count = list(np.bincount([self.rules_len[x] for x in rules], minlength = self.maxlen+1))
        prior_ChsRules= sum([log_betabin(Kn_count[i],
                                         self.patternSpace[i],self.alpha_l[i],
                                         self.beta_l[i]) for i in range(1,len(Kn_count),1)])
        likelihood_1 = log_betabin(TP, TP+FP, self.alpha_1, self.beta_1)
        likelihood_2 = log_betabin(TN, FN+TN, self.alpha_2, self.beta_2)
        return [TP, FP, TN, FN], [prior_ChsRules, likelihood_1, likelihood_2]

    def print_rules(self, rules):
        for rule in rules:
            if self.rule_explainations.get(rule) == None:
                rules_list = [self.itemNames[item] for item in self.rules[rule]]
            else:
                rules_list = self.rule_explainations[rule][0]
            reformatted_rules = self.rewrite_rules(rules_list)
            print(reformatted_rules)

    def rewrite_rules(self, rules_list):
        rewritten_rules = []
        for rule in rules_list:
            if '<' in rule:
                rule_num = rule.split('<')
                if rule_num[1] in self.attributeNames:
                    rewritten_rules.append(rule_num[0]+'<='+rule_num[1])
                else:
                    rewritten_rules.append(rule)
            else:
                rewritten_rules.append(rule)
        return rewritten_rules



    def Bayesian_patternbased(self, X, X_trans, y, RMatrix, init_rules):
        
        # |A| : min((rule_space)/2,(rule_space+beta_l-alpha_l)/2)
        self.Asize = [[min(self.patternSpace[l]/2,
                           0.5*(self.patternSpace[l]+self.beta_l[l]-self.alpha_l[l])) for l in range(self.maxlen+1)]]
        # support threshold
        self.C = [1]

        nRules = len(self.rules)
        self.maps = defaultdict(list)
        T0 = 1000
        
        rules_curr = init_rules
        rules_curr_norm = self.normalize(rules_curr)
        pt_curr = -1000000000
        # now only consider 1 chain
        # it should have been maps[chain]
        self.maps[0].append([-1, [pt_curr/3, pt_curr/3, pt_curr/3],
                             rules_curr, [self.rules[i] for i in rules_curr],
                             []])
        for ith_iter in range(self.max_iter):
            rules_new, rules_norm = self.propose(rules_curr, rules_curr_norm,
                                                 nRules, X, X_trans, y,
                                                 RMatrix)
            cfmatrix, prob = self.compute_prob(RMatrix, y, rules_new)
            T = T0**(1 - ith_iter/self.max_iter)
            pt_new = sum(prob)
            alpha = np.exp(float(pt_new - pt_curr)/T)

            if pt_new > sum(self.maps[0][-1][1]):
                print('\n** chain = {}, max at iter = {} ** \n accuracy = {}, TP = {},FP = {}, TN = {}, FN = {}\n old is {}, pt_new is {}, prior_ChsRules={}, likelihood_1 = {}, likelihood_2 = {}\n '.format(self.chains, ith_iter,(cfmatrix[0]+cfmatrix[2]+0.0)/len(y),cfmatrix[0],cfmatrix[1],cfmatrix[2],cfmatrix[3],sum(self.maps[0][-1][1])+0.1,sum(prob), prob[0], prob[1], prob[2]))
                rules_new = self.clean_rules(RMatrix, rules_new)
                self.print_rules(rules_new)
                print(rules_new)
                self.Asize.append([np.floor(min(self.Asize[-1][l], 
                                            (-pt_new + self.Lup + self.P0)/self.const_denominator[l])) 
                                            for l in range(self.maxlen+1)])
                self.const_denominator = [np.log(np.true_divide(self.patternSpace[l]+self.beta_l[l]-1,
                                                                max(1,self.Asize[-1][l]+self.alpha_l[l]-1))) 
                                                                for l in range(self.maxlen+1)]
                self.maps[0].append([ith_iter, prob, rules_new, 
                                [self.rules[i] for i in rules_new],
                                cfmatrix])
                new_supp = np.ceil(np.log(max([np.true_divide(self.patternSpace[l]-self.Asize[-1][l]+self.beta_l[l],
                                                              max(1,self.Asize[-1][l]-1+self.alpha_l[l])) 
                                                              for l in range(1,self.maxlen+1,1)])))
                self.C.append(new_supp)
                self.predicted_rules = rules_new
            if random() <= alpha:
                rules_curr_norm, rules_curr, pt_curr = rules_norm[:], rules_new[:], pt_new
        
        return self.maps[0]

    def clean_rules(self, RMatrix, rules):
        # If the rules are not
        # modified then there is no 
        # need to check the rules
        cleaned = []
        for rule in rules:
            if self.rule_explainations.get(rule) == None:
                if sum(RMatrix[:, rule]) != 0:
                    cleaned.append(rule)
            else:
                if sum(self.rule_explainations[rule][1]) != 0:
                    cleaned.append(rule)
        return cleaned


    def fit(self, X, y):
        '''Fit model with traning data

        Parameters
        ----------
        X : pandas dataframe, (n_samples, n_features)
            data

        Y : ndarray, shape (n_samples,)
            binary target
        '''
        if self.binary_input:
            X_trans = X
        else:
            X_trans = self.transform(X, neg=self.neg)
        self.set_parameters(X_trans)
        self.precompute(y)
        if ~self.binary_input:
            self.compute_cutoffs(X, y)
        self.generate_rules(X_trans, y)
        RMatrix = self.screen_rules(X_trans, y)
        # rules_explaination records the meaning of the rules
        # and their corresponding Z 
        self.rule_explainations = dict()
        if self.greedy_initilization:
            init = self.greedy_init(X, X_trans, y, RMatrix)
        else:
            init = []
        self.Bayesian_patternbased(X, X_trans, y, RMatrix, init)

    def predict(self, X):
        # Use the mined rules to predict X
        yhat = np.zeros(X.shape[0], dtype=int)

        for rule in self.predicted_rules:
            yhat_items = np.ones(X.shape[0], dtype=int)
            rule_explaination = self.rule_explainations.get(rule)
            if rule_explaination == None:
                item_meanings = []
                for item in self.rules[rule]:
                    item_meanings.append(self.itemNames[item])
            else:
                item_meanings = rule_explaination[0]
            for item_meaning in item_meanings:
                item_meaning_cat = item_meaning.split('§')
                item_meaning_num = item_meaning.split('<')
                if len(item_meaning_cat) > 1:
                    if item_meaning_cat[-1] == 'neg':
                        if item_meaning_cat[1] == '':
                            yhat_items = [len(y) != 0 for y in X[item_meaning_cat[0]].values] & yhat_items
                        else:
                            yhat_items = ([item_meaning_cat[1] not in y for y in X[item_meaning_cat[0]].values]) & yhat_items
                    else:
                        if item_meaning_cat[1] == '':
                            yhat_items = [len(y) == 0 for y in X[item_meaning_cat[0]].values] & yhat_items
                        else:
                            yhat_items = ([item_meaning_cat[1] in y for y in X[item_meaning_cat[0]].values]) & yhat_items
                else:
                    if item_meaning_num[0] in self.attributeNames:
                        yhat_items = (X[item_meaning_num[0]] < float(item_meaning_num[1])) & yhat_items
                    else:
                        yhat_items = (float(item_meaning_num[0]) <= X[item_meaning_num[1]]) & yhat_items
                print(yhat_items)
            yhat = yhat | yhat_items

        return yhat


def log_betabin(k, n, alpha, beta):
    try:
        c = math.lgamma(alpha+beta) - math.lgamma(alpha) - math.lgamma(beta)
    except:
        print('alpha = {}, beta = {}'.format(alpha, beta))
    if isinstance(k, (list, np.ndarray)):
        if len(k) != len(n):
            print('length of k in %d and length of n is %d' %(len(k), len(n)))
            raise ValueError
        lbeta = []
        for ki, ni in zip(k, n):
            lbeta.append(math.lgamma(ki+alpha) + math.lgamma(ni-ki+beta) - math.lgamma(ni+alpha+beta) + c)
        return np.array(lbeta)
    else:
        return (math.lgamma(k+alpha) + math.lgamma(n-k+beta) - math.lgamma(n+alpha+beta) + c)
                
def get_confusion(yhat, y):
    if len(yhat) != len(y):
        raise NameError('yhat has different length')
    TP = sum(np.array(yhat) & np.array(y))
    predict_pos = np.sum(yhat)
    FP = predict_pos - TP
    TN = len(y) - np.sum(y) - FP
    FN = len(yhat) - predict_pos - TN
    return TP, FP, TN, FN

def extract_rules(tree, feature_names):
    left = tree.tree_.children_left
    if left[0] == -1:
        return [sample(list(feature_names), 1)]
    right = tree.tree_.children_right
    features = [feature_names[i] for i in tree.tree_.feature]
    idx = np.where(left==-1)[0]
    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left==child)[0].item()
        else:
            parent = np.where(right==child)[0].item()
        lineage.append(features[parent])
        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
    rules = []
    for child in idx:
        rule = []
        # in case the tree is empty
        if len(left)>1:
            for node in recurse(left, right, child):
                rule.append(node)
            rules.append(rule)
        else:
            pass
    return rules

def accumulate(iterable, func=operator.add):
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def find_lt(a, x):
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    else:
        return 0

def remove_duplicates(l):
    elements = {}
    for element in l:
        elements[element] = 1
    return list(elements.keys())

def find_interval(idx1, l2):
    '''This function takes a index 
    of the list of all the elements,
    and a list of length
    of each type of elements.
    returns a idx2 for the list of length
    '''
    idx2 = 0
    tmp_sum = 0
    for i in l2:
        tmp_sum += i
        if tmp_sum >= idx1+1:
            return idx2
        else:
            idx2 += 1

