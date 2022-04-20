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
    matrix, inds, cols = X_trans
    filter_items = {}
    relations_ab = {}
    inv_relations_ab = {}
    k_as_sub = {}
    k_as_obj = {}
    cleaned_relations = set()
    cx = matrix.tocoo()
    for i, j, v in tqdm(list(zip(cx.row, cx.col, cx.data))):
        if '§' in cols[j]:
            rel, obj = cols[j].split('§')
            if rel not in relations_ab:
                relations_ab[rel]=set()
                inv_relations_ab[rel]=set()
            relations_ab[rel].add((inds[i],obj))
            inv_relations_ab[rel].add((obj,inds[i]))

            if rel not in k_as_sub:
                k_as_sub[rel] = {}
            if inds[i] not in k_as_sub[rel]:
                k_as_sub[rel][inds[i]] = set()
            k_as_sub[rel][inds[i]].add(obj)

            if rel not in k_as_obj:
                k_as_obj[rel] = {}
            if obj not in k_as_obj[rel]:
                k_as_obj[rel][obj] = set()
            k_as_obj[rel][obj].add(inds[i])
        else:
            cleaned_relations.add(cols[j])

    matrix, inds, cols = None, None, None
    gc.collect()

    cleaned_relations = [c for c in cleaned_relations if len(relations_ab[c])>=miner.support]

    for c in cleaned_relations:
        filter_items[('?a '+c+' ?b',)] = len(relations_ab[c])
        filter_items[('?b ' + c + ' ?a',)] = len(relations_ab[c])

    _pr_comb = list(itertools.combinations_with_replacement(cleaned_relations,2))


    cleaned_relations = [c for c in cleaned_relations if
                         len(relations_ab[c]) >= miner.support and c.count(':') < miner.max_rule_set - 1]
    cleaned_single_rel = [c for c in cleaned_relations if c.count(':') == 1]

    _pr = list(itertools.product(cleaned_relations, repeat=2))

    if miner.rule_complexity > 0:
        with Pool(mp.cpu_count() - 1, initializer=__init,
                  initargs=(k_as_sub, k_as_obj, relations_ab,inv_relations_ab, miner.max_rule_set, miner.support, cleaned_single_rel)) as pool:

            for r in tqdm(pool.imap_unordered(exec_f1, _pr_comb, chunksize=1000), total=len(_pr_comb)):
                for el in r:
                    filter_items[el] = r[el]

            if miner.rule_complexity > 1:
                for r in tqdm(pool.imap_unordered(exec, _pr, chunksize=1), total=len(_pr)):
                    p, cons_sub, cons_objs, ant_subs, ant_objs = r
                # = exec(p, cleaned_relations,k_as_sub, k_as_obj, relations_ab, miner.max_rule_set, miner.support)
                    for ant in cons_sub:
                        if cons_sub[ant]>= miner.support:
                            filter_items[(('?k ' + p[0] + ' ?a', '?k ' + p[1] + ' ?b'),)] = ant_subs
                            filter_items[(('?k ' + p[0] + ' ?a', '?k ' + p[1] + ' ?b'), '?a ' + ant + ' ?b',)] = cons_sub[ant]
                    for ant in cons_objs:
                        if cons_objs[ant] >= miner.support:
                            filter_items[(('?a ' + p[0] + ' ?k', '?b ' + p[1] + ' ?k'),)] = ant_objs
                            filter_items[(('?a ' + p[0] + ' ?k', '?b ' + p[1] + ' ?k'), '?a ' + ant + ' ?b',)] = cons_objs[ant]

            pool.close()
            pool.join()

    df = pd.DataFrame(list(filter_items.items()), columns=['itemsets', 'support'])
    rules = association_rules(df, metric="support", min_threshold=miner.support)
    miner.rules = rules

def __init(d1, d2, d3, d4, d5, d6, d7):
    global k_as_sub, k_as_obj, relations_ab, inv_relations_ab, rule_len,support,cleaned_relations
    k_as_sub, k_as_obj, relations_ab,inv_relations_ab, rule_len,support, cleaned_relations = d1,d2,d3,d4,d5,d6,d7

def exec_f1(p):
    filter_items = {}
    if p[0].count(':') + p[1].count(':') <= rule_len:
        if p[0] != p[1]:
            d = relations_ab[p[1]].intersection(relations_ab[p[0]])
            if len(d) >= support:
                filter_items[('?a ' + p[0] + ' ?b', '?a ' + p[1] + ' ?b',)] = len(d)

                for c in cleaned_relations:
                    if c != p[0] and c != p[1]:
                        if p[0].count(':') + p[1].count(':') + c.count(':') <= rule_len:
                            dd = d.intersection(relations_ab[c])
                            if len(dd) >= support:
                                filter_items[(('?a ' + p[0] + ' ?b', '?a ' + p[1] + ' ?b'),)] = len(d)
                                filter_items[(('?a ' + p[0] + ' ?b', '?a ' + p[1] + ' ?b'), '?a ' + c + ' ?b')] = len(
                                    dd)

        d = len(inv_relations_ab[p[1]].intersection(relations_ab[p[0]]))
        if d >= support:
            filter_items[('?b ' + p[0] + ' ?a', '?a ' + p[1] + ' ?b',)] = d
    return filter_items

def exec(p):
    cons_sub = {}
    ant_subs = -1
    cons_objs = {}
    ant_objs = -1

    if p[0].count(':') + p[1].count(':') <= rule_len - 1:
    #
        d1 = set(k_as_sub[p[0]].keys()).intersection(set(k_as_sub[p[1]].keys()))
        ant_subs_upper = sum([len(k_as_sub[p[0]][d]) * len(k_as_sub[p[1]][d]) for d in d1])

        d2 = set(k_as_obj[p[0]].keys()).intersection(set(k_as_obj[p[1]].keys()))
        ant_objs_upper = sum([len(k_as_obj[p[0]][d]) * len(k_as_obj[p[1]][d]) for d in d2])

        if ant_subs_upper >= support:
            for p3 in cleaned_relations:
                all_coms = relations_ab[p3]
                rel1_k = {k for obj in all_coms if obj[0] in k_as_obj[p[0]] for k in k_as_obj[p[0]][obj[0]]}
                if p[0]!=p[1]:
                    rel2_k = {k for obj in all_coms if obj[1] in k_as_obj[p[1]] for k in k_as_obj[p[1]][obj[1]]}
                    dd = rel1_k.intersection(rel2_k)
                else:
                    dd = rel1_k
                zz = len({(x,y) for d in dd for x in k_as_sub[p[0]][d] for y in k_as_sub[p[1]][d] if (x,y) in all_coms})
                if zz>=support:
                    if ant_subs==-1:
                        ant_subs = len({(x, y) for d in d1 for x in k_as_sub[p[0]][d] for y in k_as_sub[p[1]][d]})
                    if  ant_subs>=support:
                        cons_sub[p3] = zz

        if ant_objs_upper >= support:
            for p3 in cleaned_relations:
                all_coms = relations_ab[p3]
                rel1_k = {k for obj in all_coms if obj[0] in k_as_sub[p[0]] for k in k_as_sub[p[0]][obj[0]]}
                if p[0] != p[1]:
                    rel2_k = {k for obj in all_coms if obj[1] in k_as_sub[p[1]] for k in k_as_sub[p[1]][obj[1]]}
                    dd = rel1_k.intersection(rel2_k)
                else:
                    dd = rel1_k
                zz = len({(x, y) for d in dd for x in k_as_obj[p[0]][d] for y in k_as_obj[p[1]][d] if (x, y) in all_coms})
                if zz >= support:
                    if ant_objs==-1:
                        ant_objs = len({(x, y) for d in d2 for x in k_as_obj[p[0]][d] for y in k_as_obj[p[1]][d]})
                    if ant_objs>=support:
                        cons_objs[p3] = zz

    return p, cons_sub, cons_objs, ant_subs, ant_objs