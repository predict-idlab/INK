"""
rulemining.py file
File which contains the full mining capability using the binary INK representation.

This file is adapted from:
Bayesian Rule Set mining by Tong Wang and Peter (Zhen) Li
reference: Wang, Tong, et al. "Bayesian rule sets for interpretable classification.
Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016.
"""

import math
import random
import numpy as np
import pandas as pd
from scipy import sparse
import ink.miner.utils as utils
from ink.miner.task_agnostic_mining import agnostic_fit
from ink.miner.task_specific_mining import specific_fit

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'

np.seterr(over='ignore')
np.seterr(divide='ignore', invalid='ignore')
pd.options.mode.chained_assignment = None


class RuleSetMiner(object):
    """
    The INK RuleSetMiner.
    Class which can mine both task specific and task agnostic rules.

    :param support: Support measure, only rules with this level of support will be taken into account.
    :type support: int
    :param max_rules: Maximal number of rules which can be mined.
    :type max_rules: int
    :param max_len_rule_set: Maximal number of rules used to separate the classes during task-specific mining.
    :type max_len_rule_set: int
    :param max_iter: Maximal number of iterations used for the task-specific miner.
    :type max_iter: int
    :param chains: Maximal number of chains used for the task-specific miner.
    :type chains: int
    :param forest_size: Maximal number of forest within the classifier for the task-specific miner.
    :type forest_size: int
    :param criteria: Criteria used to screen the generated rules. Possible criteria's are precision, specificity,
                     sensitivity, mcc (matthew correlation coefficient) or cross-entropy (default).
    :type criteria: str
    :param propose_threshold: Threshold used to propose new combinations of possible rules for the task-specific mining.
    :type propose_threshold: int
    :param verbose: Parameter to show tqdm tracker (default False).
    :type: bool
    """
    def __init__(self, support=10, max_rules=10e13, max_len_rule_set=5, max_iter=10, chains=1000, forest_size=1000,
                 criteria='precision', rule_complexity = 2, propose_threshold=0.1, verbose=False):
        self.max_rules = max_rules
        self.max_iter = max_iter
        self.chains = chains
        self.support = support
        self.max_rule_set = max_len_rule_set
        self.verbose = verbose
        self.alpha_1 = 100
        self.beta_1 = 1
        self.alpha_2 = 100
        self.beta_2 = 1
        self.alpha_l = None
        self.beta_l = None
        self.propose_threshold = propose_threshold
        self.forest_size = forest_size
        self.predicted_rules = []
        self.dct_check = {}
        self.criteria = criteria
        self.attributeNames = None
        self.itemNames = None
        self.rule_explanations = None
        self.rules_len = None
        self.P0 = None
        self.const_denominator = None
        self.Lup = None
        self.patternSpace = []
        self.rules = []
        self.rule_complexity = rule_complexity

    def fit(self, data, label=None):
        """
        Fit function to train the classifier or generate agnostic rules
        :param data: Tuple value containing 1) a sparse binary representation, 2) list of indices, 3) column features.
        :type data: tuple
        :param label: List containing the labels for each index (task-specific) or None (task-agnostic)
        :return: Rules
        """
        if label is not None:
            return specific_fit(self, data, label)
        else:
            return agnostic_fit(self, data)

    def predict(self, data):
        """
        Predict function used to predict new data against the learned task-specific rules.
        :param data: Tuple value containing 1) a sparse binary representation, 2) list of indices, 3) column features.
        :type data: tuple
        :return: Predicted labels
        :rtype: list
        """
        df = pd.DataFrame(data[0].todense())
        df.index = data[1]
        df.columns = data[2]
        X = df.astype('bool')

        # replace this with multiprocessing code
        yhat = np.zeros(X.shape[0], dtype=int)
        for rule in self.predicted_rules:
            yhat_items = np.ones(X.shape[0], dtype=int)
            for item in self.rules[rule]:
                if self.itemNames[item] in X.columns:
                    yhat_items = X[self.itemNames[item]].values & yhat_items
                else:
                    if self.itemNames[item].startswith('count.'):
                        if '<' in self.itemNames[item]:
                            yhat_items = np.ones(X.shape[0], dtype=int) & yhat_items
                        else:
                            yhat_items = np.zeros(X.shape[0], dtype=int) & yhat_items
                    else:
                        yhat_items = np.zeros(X.shape[0], dtype=int) & yhat_items
                if self.verbose:
                    print(yhat_items)
            yhat = yhat | yhat_items
        return yhat

    def print_rules(self, rules):
        """
        Function to represent the rules in a human-readable format.
        :param rules: Output generated from the task-specific fit function
        :type rules: list
        :return:
        """
        for rule in rules:
            if self.rule_explanations.get(rule) is None:
                rules_list = [self.itemNames[item] for item in self.rules[rule]]
            else:
                rules_list = self.rule_explanations[rule][0]
            reformatted_rules = utils.rewrite_rules(rules_list, self.attributeNames)
            print(reformatted_rules)

    def set_parameters(self, X):
        """
        Function to set some initial parameters based on the data.
        :param X: Tuple value containing 1) a sparse binary representation, 2) list of indices, 3) column features.
        :type X: tuple
        :return:
        """
        # number of possible rules, i.e. rule space italic(A) prior
        self.patternSpace = np.ones(self.max_rule_set + 1)
        # This patternSpace is an approximation
        # because the original code allows
        # the following situation, take tic-tac-toe
        # 1_O == 1 and 1_O_neg == 1, which is impossible
        numAttributes = len(X[2])
        for i in range(1, self.max_rule_set + 1):
            tmp = 1
            for j in range(numAttributes - i + 1, numAttributes + 1):
                tmp *= j
            self.patternSpace[i] = tmp / math.factorial(i)

        if self.alpha_l is None:
            self.alpha_l = [1 for _ in range(self.max_rule_set + 1)]
        if self.beta_l is None:
            self.beta_l = [(self.patternSpace[i] * 100 + 1) for i in range(self.max_rule_set + 1)]

    def precompute(self, y):
        """
        Precompute values based on the given labels.
        :param y: List of labels.
        :return:
        """
        TP, FP, TN, FN = sum(y), 0, len(y) - sum(y), 0
        # self.Lup : p(S|A;alpha_+,beta_+,alpha_-,beta_-)
        # conference paper formula(6)
        self.Lup = (utils.log_betabin(TP, TP + FP, self.alpha_1, self.beta_1)
                    + utils.log_betabin(TN, FN + TN, self.alpha_2, self.beta_2))
        # self.const_denominator : log((|Al|+beta_l-1)/(alpha_l+|Al|-1))
        # conference paper formula(9) denominator
        self.const_denominator = [np.log((self.patternSpace[i] + self.beta_l[i] - 1)
                                         / (self.patternSpace[i] + self.alpha_l[i] - 1))
                                  for i in range(self.max_rule_set + 1)]
        Kn_count = np.zeros(self.max_rule_set + 1, dtype=int)
        # P0 : maximum prior
        # Ml=0, |Al|= rule space
        # conference paper formula(3)
        # because of log property, + is *
        self.P0 = sum([utils.log_betabin(Kn_count[i], self.patternSpace[i], self.alpha_l[i],
                                         self.beta_l[i]) for i in range(1, self.max_rule_set + 1)])

    def screen_rules(self, X_trans, y):
        """
        Function to pre_screen the generated rules based on the enabled criteria
        :param X_trans: Binary data frame.
        :param y: Label list
        :return: RMatrix
        """
        tmp_rules_len = [len(rule) for rule in self.rules]
        ruleMatrix = np.zeros((len(self.rules), len(X_trans.columns)), dtype=int)
        for i, rule in enumerate(self.rules):
            for j in rule:
                ruleMatrix[i][j - 1] = 1
        ruleMatrix = sparse.csc_matrix(ruleMatrix.transpose())

        mat = (sparse.csc_matrix(X_trans) * ruleMatrix).todense()
        # Z is the matrix for data points covered by rules
        Z = (mat == tmp_rules_len)
        Zpos = Z[np.where(y > 0)]

        # TP for each rule
        TP = np.asarray(np.sum(Zpos, axis=0))[0]
        # supp is threshold percentile of how TP a rule is
        supp_select = np.where(TP >= self.support * sum(y) / 100.0)[0]

        if len(supp_select) <= self.max_rules:
            self.rules = np.asarray(self.rules)[supp_select]
            RMatrix = np.array(Z[:, supp_select])

            self.rules_len = [len(rule) for rule in self.rules]
        else:
            FP = np.array(np.sum(Z, axis=0))[0] - TP
            TN = len(y) - np.sum(y) - FP
            FN = np.sum(y) - TP
            p1 = TP.astype(float) / (TP + FP)
            p2 = FN.astype(float) / (FN + TN)
            pp = (TP + FP).astype(float) / (TP + FP + TN + FN)

            if self.criteria == 'precision':
                select = np.argsort(p1[supp_select])[::-1][:self.max_rules].tolist()
            elif self.criteria == 'specificity':
                p3 = TN.astype(float) / (TN + FP)
                select = np.argsort(p3[supp_select])[::-1][:self.max_rules].tolist()
            elif self.criteria == 'sensitivity':
                p4 = TP.astype(float) / (TP + FN)
                select = np.argsort(p4[supp_select])[::-1][:self.max_rules].tolist()
            elif self.criteria == 'mcc':
                p5 = (2*TP.astype(float)) / (2*TP.astype(float) + FP + FN)
                select = np.argsort(p5[supp_select])[::-1][:self.max_rules].tolist()
            else:
                cond_entropy = (-pp * (p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
                                - (1 - pp) * (p2 * np.log(p2)
                                              + (1 - p2) * np.log(1 - p2)))
                cond_entropy[p1 * (1 - p1) == 0] = (-((1 - pp) * (p2 * np.log(p2)
                                                                  + (1 - p2) * np.log(1 - p2)))[p1 * (1 - p1) == 0])
                cond_entropy[p2 * (1 - p2) == 0] = (-(pp * (p1 * np.log(p1)
                                                            + (1 - p1) * np.log(1 - p1)))[p2 * (1 - p2) == 0])
                cond_entropy[p1 * (1 - p1) * p2 * (1 - p2) == 0] = 0
                pos = (TP + FN).astype(float) / (TP + FP + TN + FN)
                info = - pos * np.log(pos) - (1 - pos) * np.log(1 - pos)
                info[np.where((pos == 1) | (pos == 0))[0]] = 0
                IGR = (info - cond_entropy) / info
                IGR[np.where(info == 0)[0]] = 0
                select = np.argsort(IGR[supp_select])[::-1][:self.max_rules].tolist()
            ind = list(supp_select[select])
            self.rules = [self.rules[i] for i in ind]
            RMatrix = np.array(Z[:, ind])
            self.rules_len = [len(rule) for rule in self.rules]
        return RMatrix

    def __normalize(self, rules_new):
        try:
            rules_len = [len(self.rules[index]) for index in rules_new]
            rules = [rules_new[i] for i in np.argsort(rules_len)[::-1][:len(rules_len)]]
            p1 = 0
            while p1 < len(rules):
                for p2 in range(p1 + 1, len(rules), 1):
                    if set(self.rules[rules[p2]]).issubset(set(self.rules[rules[p1]])):
                        rules.remove(rules[p1])
                        p1 -= 1
                        break
                p1 += 1
            return rules[:]
        except (ValueError, Exception):
            return rules_new[:]

    def __find_rules_z(self, RMatrix, rules):
        if len(rules) == 0:
            return np.zeros(RMatrix.shape[0], dtype=int)
        Z = np.zeros(RMatrix.shape[0], dtype=int)
        for rule in rules:
            if self.rule_explanations.get(rule) is None:
                Z = RMatrix[:, rule] + Z
            else:
                Z = self.rule_explanations[rule][1] + Z
        Z = Z > 0
        return Z

    def __propose(self, rules_curr, rules_norm, RMatrix, Y, q):
        nRules = len(self.rules)
        Yhat = (np.sum(RMatrix[:, rules_curr], axis=1) > 0).astype(int)
        incorr = np.where(Y != Yhat)[0]
        N = len(rules_curr)
        if len(incorr) == 0:
            ex = None
            move = ['clean']
            # it means the HBOA correctly classified all points but there could be redundant patterns,
            # so cleaning is needed
        else:
            ex = random.sample(list(incorr), 1)[0]
            t = np.random.random()
            if Y[ex] == 1 or N == 1:
                if t < 1.0 / 2 or N == 1:
                    move = ['add']  # action: add
                else:
                    move = ['cut', 'add']  # action: replace
            else:
                if t < 1.0 / 2:
                    move = ['cut']  # action: cut
                else:
                    move = ['cut', 'add']  # action: replace
        if move[0] == 'cut':
            """ cut """
            if np.random.random() < q:
                candidate = list(set(np.where(RMatrix[ex, :] == 1)[0]).intersection(rules_curr))
                if len(candidate) == 0:
                    candidate = rules_curr
                cut_rule = random.sample(list(candidate), 1)[0]
            else:
                p = []
                all_sum = np.sum(RMatrix[:, rules_curr], axis=1)
                for index, rule in enumerate(rules_curr):
                    Yhat = ((all_sum - np.array(RMatrix[:, rule])) > 0).astype(int)
                    TP, FP, TN, FN = utils.get_confusion(Yhat, Y)
                    p.append(TP.astype(float) / (TP + FP + 1))

                p = [x - min(p) for x in p]
                p = np.exp(p)
                p = np.insert(p, 0, 0)
                p = np.array(list(utils.accumulate(p)))
                if p[-1] == 0:
                    index = random.sample(range(len(rules_curr)), 1)[0]
                else:
                    p = p / p[-1]
                    # here
                    index = utils.find_lt(p, np.random.random())
                cut_rule = rules_curr[index]
            rules_curr.remove(cut_rule)
            rules_norm = self.__normalize(rules_curr)
            move.remove('cut')

        if len(move) > 0 and move[0] == 'add':
            """ add """
            if np.random.random() < q:
                add_rule = random.sample(range(nRules), 1)[0]
            else:
                Yhat_neg_index = list(np.where(np.sum(RMatrix[:, rules_curr], axis=1) < 1)[0])
                mat = np.multiply(RMatrix[Yhat_neg_index, :].transpose(), Y[Yhat_neg_index])
                # TP = np.array(np.sum(mat,axis = 0).tolist()[0])
                TP = np.sum(mat, axis=1)
                FP = np.array((np.sum(RMatrix[Yhat_neg_index, :], axis=0) - TP))
                # TN = np.sum(Y[Yhat_neg_index] == 0) - FP
                # FN = sum(Y[Yhat_neg_index]) - TP
                p = (TP.astype(float) / (TP + FP + 1))
                p[rules_curr] = 0
                add_rule = random.sample(list(np.where(p == max(p))[0]), 1)[0]
            if add_rule not in rules_curr:
                rules_curr.append(add_rule)
                rules_norm = self.__normalize(rules_curr)

        if len(move) > 0 and move[0] == 'clean':
            remove = []
            for i, rule in enumerate(rules_norm):
                Yhat = (np.sum(
                    RMatrix[:, [rule for j, rule in enumerate(rules_norm) if (j != i and j not in remove)]],
                    axis=1) > 0).astype(int)
                TP, FP, TN, FN = utils.get_confusion(Yhat, Y)
                if TP + FP == 0:
                    remove.append(i)
            for x in remove:
                if x in rules_norm:
                    rules_norm.remove(x)
            return rules_curr, rules_norm
        return rules_curr, rules_norm

    def __compute_prob(self, rules, RMatrix, Y):
        Yhat = (np.sum(RMatrix[:, rules], axis=1) > 0).astype(int)
        TP, FP, TN, FN = utils.get_confusion(Yhat, Y)
        Kn_count = list(np.bincount([self.rules_len[x] for x in rules], minlength=self.max_rule_set + 1))
        prior_ChsRules = sum([utils.log_betabin(Kn_count[i], self.patternSpace[i], self.alpha_l[i], self.beta_l[i])
                              for i in range(1, len(Kn_count), 1)])
        likelihood_1 = utils.log_betabin(TP, TP + FP, self.alpha_1, self.beta_1)
        likelihood_2 = utils.log_betabin(TN, FN + TN, self.alpha_2, self.beta_2)
        return [TP, FP, TN, FN], [prior_ChsRules, likelihood_1, likelihood_2]

    def exec_chain(self, t):
        """
        Function to execute chaining in parallel.
        :param t: Tuple with number of rules, split, the RMatrix, y, T0 and chain indicator
        :type t: tuple
        :return: Chaining results
        :rtype: list
        """
        nRules, split, RMatrix, y, T0, chain = t
        # random.seed()
        # np.random.seed()
        lst = []
        N = random.sample(range(1, min(8, nRules), 1), 1)[0]
        rules_curr = random.sample(range(nRules), N)
        rules_curr_norm = self.__normalize(rules_curr)
        pt_curr = -100000000000
        lst.append(
            [-1, [pt_curr / 3, pt_curr / 3, pt_curr / 3], rules_curr, [self.rules[i] for i in rules_curr]])

        for i in range(self.max_iter):
            if i >= split:
                p = np.array(range(1 + len(lst)))
                p = np.array(list(utils.accumulate(p)))
                p = p / p[-1]
                index = utils.find_lt(p, np.random.random())
                rules_curr = lst[index][2].copy()
                rules_curr_norm = lst[index][2].copy()
            rules_new, rules_norm = self.__propose(rules_curr.copy(), rules_curr_norm.copy(), RMatrix, y,
                                                   self.propose_threshold)
            cfmatrix, prob = self.__compute_prob(rules_new, RMatrix, y)
            T = T0 ** (1 - i / self.max_iter)
            pt_new = sum(prob)
            alpha = np.exp(float(pt_new - pt_curr) / T)

            if pt_new > sum(lst[-1][1]):
                lst.append([i, prob, rules_new, [self.rules[i] for i in rules_new], cfmatrix])
            if np.random.random() <= alpha:
                rules_curr_norm, rules_curr, pt_curr = rules_norm.copy(), rules_new.copy(), pt_new

        return lst
