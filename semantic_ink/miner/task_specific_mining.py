"""
task_specific_mining.py file.
Defines the functions and classes to mine task specific rules.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import semantic_ink.miner.utils as utils
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'


def specific_fit(miner, X_trans, y):
    """
    Function to mine task-specific rules
    :param miner: instance of the RuleSetMiner
    :param X_trans: Tuple value containing 1) a sparse binary representation, 2) list of indices, 3) column features.
    :type X_trans: tuple
    :param y: List containing the labels for each index
    :type y: list
    :return: Rules
    """
    miner.set_parameters(X_trans)
    df = pd.DataFrame(X_trans[0].todense())
    df.index = X_trans[1]
    df.columns = X_trans[2]
    X_trans = df.astype('bool')
    miner.precompute(y)
    __specific_rules(miner, X_trans, y)
    miner.attributeNames = X_trans.columns

    RMatrix = miner.screen_rules(X_trans, y)
    miner.rule_explanations = dict()
    return __bayesian_patternbased(miner, y, RMatrix)


def __calc_estimators(t):
    n_estimators, length, X_trans, y, items = t
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=length)
    clf.fit(X_trans, y)
    sub_rules = []
    for n in range(n_estimators):
        sub_rules.extend(utils.extract_rules(clf.estimators_[n], items))
    return sub_rules


def __specific_rules(miner, X_trans, y):
    itemNames = dict()

    for i, item in enumerate(X_trans.columns):
        itemNames[i + 1] = item

    miner.itemNames = itemNames

    items = np.arange(1, len(X_trans.columns) + 1)

    rules = []

    if miner.verbose:
        print("Create classification forests")
    with Pool(mp.cpu_count() - 1) as pool:
        seq = [(miner.forest_size * length, length, X_trans, y, items) for length in range(1, miner.max_rule_set + 1)]
        res = list(tqdm(pool.imap_unordered(__calc_estimators, seq, chunksize=1), total=len(seq),
                        disable=not miner.verbose))

    for r in res:
        rules.extend(r)

    miner.rules = rules


def __bayesian_patternbased(miner, y, RMatrix):
    nRules = len(miner.rules)
    miner.rules_len = [len(rule) for rule in miner.rules]
    T0 = 1000
    split = 0.7 * miner.max_iter

    seq = [(nRules, split, RMatrix, y, T0, c) for c in range(miner.chains)]
    if miner.verbose:
        print("Chaining")
    with Pool(mp.cpu_count()-1) as pool:
        maps = list(tqdm(pool.imap_unordered(miner.exec_chain, seq, chunksize=8), total=miner.chains,
                         disable=not miner.verbose))

    pt_max = [sum(maps[chain][-1][1]) for chain in range(miner.chains)]
    index = pt_max.index(max(pt_max))

    miner.cfmatrix = maps[index][-1][-1]
    TP = miner.cfmatrix[0]
    FP = miner.cfmatrix[1]
    TN = miner.cfmatrix[2]
    FN = miner.cfmatrix[3]
    acc = (TP + TN) / (TP + FP + TN + FN)

    miner.predicted_rules = maps[index][-1][2]
    return acc, maps[index][-1][2]
