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
import mlxtend
__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'

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

from math import factorial
def nPr(n, r):
    if n-r>0:
        return int(factorial(n) / factorial(n - r))
    else:
        return 0

from collections import defaultdict
def __agnostic_rules(miner, X_trans):
    global  relations_ab, rule_len, support, cleaned_relations #k_as_sub, k_as_obj,
    support = miner.support
    rule_len = miner.max_rule_set
    matrix, inds, cols = X_trans
    filter_items = {}
    relations_ab = {}
    #k_as_sub = {}
    #k_as_obj = {}
    cleaned_relations = set()
    cx = matrix.tocoo()
    sum_cols = matrix.sum(axis=0).tolist()[0]
    mapper = set()
    col_mapper = {}
    for c in tqdm(range(len(cols))):
        if '§' in cols[c]:
            rel, obj = cols[c].split('§')
            mapper.add(rel)
            mapper.add(obj)
            if rel not in col_mapper:
                col_mapper[rel]=0
            col_mapper[rel]+=sum_cols[c]
    for i in tqdm(inds):
        mapper.add(i)

    mapper_dct = {k: v for v, k in enumerate(mapper)}
    mapper_dct_inv = {v: k for v, k in enumerate(mapper)}

    for i, j, v in tqdm(list(zip(cx.row, cx.col, cx.data))):
        if '§' in cols[j]:
            rel, obj = cols[j].split('§')
            if col_mapper[rel]>=support:
                rel = mapper_dct[rel]
                obj = mapper_dct[obj]
                subj = mapper_dct[inds[i]]
                if rel not in relations_ab:
                    relations_ab[rel]=set()
                relations_ab[rel].add((subj,obj))

        else:
            if col_mapper[cols[j]] >= support:
                cleaned_relations.add(mapper_dct[cols[j]])

    matrix, inds, cols = None, None, None
    mapper_dct = None
    col_mapper = None
    sum_cols = None
    gc.collect()

    cleaned_relations = [c for c in cleaned_relations if len(relations_ab[c])>=miner.support]

    for c in cleaned_relations:
        filter_items[('?a '+mapper_dct_inv[c]+' ?b',)] = len(relations_ab[c])
        filter_items[('?b ' + mapper_dct_inv[c] + ' ?a',)] = len(relations_ab[c])

    _pr_comb = itertools.combinations_with_replacement(cleaned_relations,2)
    _pr_comb = [(p,(mapper_dct_inv[p[0]],mapper_dct_inv[p[1]])) for p in _pr_comb if mapper_dct_inv[p[0]].count(':') + mapper_dct_inv[p[1]].count(':') <= rule_len]

    cleaned_relations = [c for c in cleaned_relations if
                         len(relations_ab[c]) >= miner.support and mapper_dct_inv[c].count(':') < miner.max_rule_set - 1]
    cleaned_single_rel = [(c,mapper_dct_inv[c]) for c in cleaned_relations if mapper_dct_inv[c].count(':') == 1]
    if miner.rule_complexity > 0:
        with Pool(mp.cpu_count()-1, initializer=__init,
                  initargs=(relations_ab, miner.max_rule_set, miner.support, cleaned_relations, cleaned_single_rel)) as pool:

            for r in tqdm(pool.imap_unordered(exec_f1, _pr_comb, chunksize=1000), total=len(_pr_comb)):
                for el in r:
                    filter_items[el] = r[el]

            _pr_comb = None
            cleaned_single_rel = None
            for p in relations_ab:
                relations_ab[p] = list(relations_ab[p])
            gc.collect()

            _pr = itertools.product(cleaned_relations, repeat=2)
            _pr = [p for p in _pr if mapper_dct_inv[p[0]].count(':') + mapper_dct_inv[p[1]].count(':')  <= rule_len - 1]


            for r in tqdm(pool.imap_unordered(exec, _pr, chunksize=1), total=len(_pr)):
                p, cons_sub, cons_objs, ant_subs, ant_objs = r
                for ant in cons_sub:
                    if cons_sub[ant]>= miner.support:
                        filter_items[(('?k ' + mapper_dct_inv[p[0]] + ' ?a', '?k ' + mapper_dct_inv[p[1]] + ' ?b'),)] = ant_subs
                        filter_items[(('?k ' + mapper_dct_inv[p[0]] + ' ?a', '?k ' + mapper_dct_inv[p[1]] + ' ?b'), '?a ' + mapper_dct_inv[ant] + ' ?b',)] = cons_sub[ant]
                for ant in cons_objs:
                    if cons_objs[ant] >= miner.support:
                        filter_items[(('?a ' + mapper_dct_inv[p[0]] + ' ?k', '?b ' + mapper_dct_inv[p[1]] + ' ?k'),)] = ant_objs
                        filter_items[(('?a ' + mapper_dct_inv[p[0]] + ' ?k', '?b ' + mapper_dct_inv[p[1]] + ' ?k'), '?a ' + mapper_dct_inv[ant] + ' ?b',)] = cons_objs[ant]

            pool.close()
            pool.terminate()

    df = pd.DataFrame(list(filter_items.items()), columns=['itemsets', 'support'])
    rules = association_rules(df, metric="support", min_threshold=miner.support)
    miner.rules = rules

def __init(d1, d2, d3, d4, d5):
    global relations_ab, rule_len,support,cleaned_relations, cleaned_single_rel
    relations_ab, rule_len,support, cleaned_relations, cleaned_single_rel = d1,d2,d3,d4,d5


def exec_f1(p):
    p,e = p
    filter_items = {}
    if p[0] != p[1]:
        d = relations_ab[p[1]].intersection(relations_ab[p[0]])
        if len(d) >= support:
            filter_items[('?a ' + e[0] + ' ?b', '?a ' + e[1] + ' ?b',)] = len(d)

            for x,c in cleaned_single_rel:
                if c != e[0] and c != e[1]:
                    if e[0].count(':') + e[1].count(':') + c.count(':') <= rule_len:
                        dd = d.intersection(relations_ab[x])
                        if len(dd) >= support:
                            filter_items[(('?a ' + e[0] + ' ?b', '?a ' + e[1] + ' ?b'),)] = len(d)
                            filter_items[(('?a ' + e[0] + ' ?b', '?a ' + e[1] + ' ?b'), '?a ' + c + ' ?b')] = len(
                                dd)

    d = len({x[::-1] for x in relations_ab[p[1]]}.intersection(relations_ab[p[0]]))
    if d >= support:
        filter_items[('?b ' + e[0] + ' ?a', '?a ' + e[1] + ' ?b',)] = d
    return filter_items


def exec(p):
    cons_sub = {}
    cons_objs = {}

    k_as_sub = {p[0]:defaultdict(set), p[1]:defaultdict(set)}
    k_as_obj = {p[0]: defaultdict(set), p[1]: defaultdict(set)}
    for c in relations_ab[p[0]]:
        k_as_sub[p[0]][c[0]].add(c[1])
        k_as_obj[p[0]][c[1]].add(c[0])

    if p[0]!=p[1]:
        for c in relations_ab[p[1]]:
            k_as_sub[p[1]][c[0]].add(c[1])
            k_as_obj[p[1]][c[1]].add(c[0])
    else:
        k_as_sub[p[1]] =  k_as_sub[p[0]]
        k_as_obj[p[1]] = k_as_obj[p[0]]

    a = set()
    b = set()
    ant_subs = 0
    order = {}

    for d in set(k_as_sub[p[0]].keys()).intersection(k_as_sub[p[1]].keys()):
        order[d] = len(k_as_sub[p[0]][d]) * len(k_as_sub[p[1]][d])

    for d in dict(sorted(order.items(), key=lambda item: -item[1])):
        ant_subs += order[d]
        if len(k_as_sub[p[0]][d].intersection(a))>0:
            if len(k_as_sub[p[1]][d].intersection(b))>0:
                ant_subs -= len(k_as_sub[p[0]][d].intersection(a))*len(k_as_sub[p[1]][d].intersection(b))
            else:
                ant_subs -= len(k_as_sub[p[0]][d].intersection(a))*len(k_as_sub[p[1]][d])
        else:
            if len(k_as_sub[p[1]][d].intersection(b)) > 0:
                ant_subs -= len(k_as_sub[p[0]][d]) * len(k_as_sub[p[1]][d].intersection(b))


        a.update(k_as_sub[p[0]][d])
        b.update(k_as_sub[p[1]][d])

    a = set()
    b = set()
    ant_objs = 0
    order = {}
    for d in set(k_as_obj[p[0]].keys()).intersection(k_as_obj[p[1]].keys()):
        order[d] = len(k_as_obj[p[0]][d]) * len(k_as_obj[p[1]][d])

    for d in dict(sorted(order.items(), key=lambda item: -item[1])):
        ant_objs += order[d]
        if len(k_as_obj[p[0]][d].intersection(a)) > 0:
            if len(k_as_obj[p[1]][d].intersection(b)) > 0:
                ant_objs -= len(k_as_obj[p[0]][d].intersection(a)) * len(k_as_obj[p[1]][d].intersection(b))
            else:
                ant_objs -= len(k_as_obj[p[0]][d].intersection(a)) * len(k_as_obj[p[1]][d])
        else:
            if len(k_as_obj[p[1]][d].intersection(b)) > 0:
                ant_objs -= len(k_as_obj[p[0]][d]) * len(k_as_obj[p[1]][d].intersection(b))


        a.update(k_as_obj[p[0]][d])
        b.update(k_as_obj[p[1]][d])

    for p3 in cleaned_relations:
        if ant_subs>=support:
            cons_sub[p3] = len(set([(k[0],k[1]) for k in relations_ab[p3] if k[0] in k_as_obj[p[0]] and k[1] in k_as_obj[p[1]] and len(k_as_obj[p[0]][k[0]].intersection(k_as_obj[p[1]][k[1]]))>0]))
        if ant_objs>=support:
            cons_objs[p3] = len(set([(k[0],k[1]) for k in relations_ab[p3] if k[0] in k_as_sub[p[0]] and k[1] in k_as_sub[p[1]] and len(k_as_sub[p[0]][k[0]].intersection(k_as_sub[p[1]][k[1]]))>0]))
    return p, cons_sub, cons_objs, ant_subs, ant_objs