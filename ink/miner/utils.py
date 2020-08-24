"""
utils.py file.
"""

import math
import random
import operator
import numpy as np
from bisect import bisect_left

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'


def log_betabin(k, n, alpha, beta):
    """
    Log_betabin function.
    """
    try:
        c = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    except (ValueError, Exception):
        c = np.inf
        print('alpha = {}, beta = {}'.format(alpha, beta))
    if isinstance(k, (list, np.ndarray)):
        if len(k) != len(n):
            print('length of k in %d and length of n is %d' % (len(k), len(n)))
            raise ValueError
        lbeta = []
        for ki, ni in zip(k, n):
            lbeta.append(math.lgamma(ki + alpha) + math.lgamma(ni - ki + beta) - math.lgamma(ni + alpha + beta) + c)
        return np.array(lbeta)
    else:
        return math.lgamma(k + alpha) + math.lgamma(n - k + beta) - math.lgamma(n + alpha + beta) + c


def get_confusion(yhat, y):
    """
    Confusion matrix function.
    """
    if len(yhat) != len(y):
        raise NameError('yhat has different length')
    TP = sum(np.array(yhat) & np.array(y))
    predict_pos = np.sum(yhat)
    FP = predict_pos - TP
    TN = len(y) - np.sum(y) - FP
    FN = len(yhat) - predict_pos - TN
    return TP, FP, TN, FN


def rewrite_rules(rules_list, attributeNames):
    """
    Rule rewrite function.
    """
    rewritten_rules = []
    for rule in rules_list:
        if '<' in rule:
            rule_num = rule.split('<')
            if rule_num[1] in attributeNames:
                rewritten_rules.append(rule_num[0] + '<=' + rule_num[1])
            else:
                rewritten_rules.append(rule)
        else:
            rewritten_rules.append(rule)
    return rewritten_rules


def extract_rules(tree, feature_names):
    """
    Rule extraction function.
    """
    left = tree.tree_.children_left
    if left[0] == -1:
        return [random.sample(list(feature_names), 1)]
    right = tree.tree_.children_right
    features = [feature_names[i] for i in tree.tree_.feature]
    idx = np.where(left == -1)[0]

    def __recurse(left_, right_, child_, lineage=None):
        if lineage is None:
            lineage = []
        if child_ in left_:
            parent = np.where(left_ == child_)[0].item()
        else:
            parent = np.where(right_ == child_)[0].item()
        lineage.append(features[parent])
        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return __recurse(left_, right_, parent, lineage)

    rules = []
    for child in idx:
        rule = []
        # in case the tree is empty
        if len(left) > 1:
            for node in __recurse(left, right, child):
                rule.append(node)
            rules.append(rule)
        else:
            pass
    return rules


def accumulate(iterable, func=operator.add):
    """
    Accumulate function.
    """
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


def find_lt(a, x):
    """
    The find_lta function.
    """
    i = bisect_left(a, x)
    if i:
        return int(i - 1)
    else:
        return 0


def remove_duplicates(el):
    """
    Remove duplicates function.
    """
    elements = {}
    for element in el:
        elements[element] = 1
    return list(elements.keys())


def find_interval(idx1, l2):
    """
    Find interval function.
    """
    idx2 = 0
    tmp_sum = 0
    for i in l2:
        tmp_sum += i
        if tmp_sum >= idx1 + 1:
            return idx2
        else:
            idx2 += 1
