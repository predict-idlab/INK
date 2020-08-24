"""
task_agnostic_mining.py file.
Defines the functions and classes to mine task agnostic rules.
"""
import gc
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
import multiprocessing as mp
from functools import lru_cache
from multiprocessing import Pool
from mlxtend.frequent_patterns import association_rules

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'

global n_dummies, filter_items, itemNames


def agnostic_fit(miner, X_trans):
    """
    Function to mine task-agnostic rules
    :param miner: instance of the RuleSetMiner
    :param X_trans: Tuple value containing 1) a sparse binary representation, 2) list of indices, 3) column features.
    :type X_trans: tuple
    :return: Rules
    """
    miner.set_parameters(X_trans)
    __agnostic_rules(miner, X_trans)
    miner.attributeNames = X_trans[2]
    return miner.rules


def __agnostic_rules(miner, X_trans):
    global itemNames, n_dummies, filter_items
    itemNames = dict()
    d = np.ones(len(X_trans[1]), dtype=bool)
    index_df = sparse.spdiags(d, False, d.size, d.size)
    index_df = sparse.hstack((index_df, np.ones(d.size, dtype=bool)[:, None]), dtype=bool)
    # TODO: fix IND
    cols = X_trans[2] + ['ind§' + x[1:-1] for x in X_trans[1]] + ['ind']

    n_dummies = sparse.hstack([X_trans[0], index_df], dtype=bool, format='csc')

    for i, item in enumerate(cols):
        itemNames[item] = i

    miner.itemNames = itemNames

    unique_vals = {}
    unique_sets = []
    for col in cols:
        c = col.split('§')
        if c[-1] not in unique_vals:
            unique_vals[c[-1]] = list()

        unique_sets.append((c[0], col))
        unique_vals[c[-1]].append(c[0])

    @lru_cache(maxsize=100000)
    def __get_column(it):
        return n_dummies.getcol(it)

    filter_items = {}
    if miner.verbose:
        print("Create filter items")
    for c, t in tqdm(unique_sets, disable=not miner.verbose):
        if (c,) not in filter_items:
            filter_items[(c,)] = 0
        if c != t:
            filter_items[(c,)] += __get_column(itemNames[t]).getnnz()

    gc.collect()
    if miner.verbose:
        print("Create item sets")
    with Pool(mp.cpu_count()-1, initializer=__init, initargs=(n_dummies, filter_items, itemNames)) as pool:
        seq = [(k, unique_vals[k], miner.support) for k in unique_vals]
        d = pool.imap_unordered(__proc, seq, chunksize=1)
        for res in tqdm(d, total=len(seq), disable=not miner.verbose):
            for m in res:
                if (m[0], m[1]) not in filter_items:
                    filter_items[(m[0], m[1])] = 0
                filter_items[(m[0], m[1])] += m[2]

    n_dummies = None

    df = pd.DataFrame(list(filter_items.items()), columns=['itemsets', 'support'])
    rules = association_rules(df, metric="support", min_threshold=miner.support)
    miner.rules = rules


def __init(d1, d2, d3):
    global n_dummies, filter_items, itemNames
    n_dummies, filter_items, itemNames = d1, d2, d3


def __proc(t):
    global n_dummies, filter_items, itemNames
    k, uk, support = t
    res = []
    if len(uk) > 1:
        combinations = list(itertools.combinations(uk, 2))
        for comb in combinations:
            if filter_items[(comb[0],)] >= support and filter_items[(comb[1],)] >= support:
                try:
                    ant = n_dummies.getcol(itemNames[comb[0] + '§' + k])
                    cons = n_dummies.getcol(itemNames[comb[1] + '§' + k])
                    value = (ant.multiply(cons)).getnnz()
                    if value > 0:
                        res.append((comb[0], comb[1], value))
                except (ValueError, Exception):
                    # TODO: this try except is requited for comb+k which are not in itemNames, handle with care
                    var = None
                    return var
    return res
