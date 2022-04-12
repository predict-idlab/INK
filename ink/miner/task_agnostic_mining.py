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

#global n_dummies, filter_items, itemNames

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
    # for p in tqdm(_pr):
    #     if p[0].count(':')+p[1].count(':')<=miner.max_rule_set:
    #         if p[0] != p[1]:
    #             d = relations_ab[p[1]].intersection(relations_ab[p[0]])
    #             if len(d) >=miner.support:
    #                 filter_items[('?a '+p[0]+' ?b','?a '+p[1]+' ?b',)] = len(d)
    #
    #                 for c in cleaned_single_rel:
    #                     if c!=p[0] and c!=p[1]:
    #                         if p[0].count(':') + p[1].count(':') + c.count(':') <= miner.max_rule_set:
    #                             dd = d.intersection(relations_ab[c])
    #                             if len(dd) >= miner.support:
    #                                 filter_items[(('?a ' + p[0] + ' ?b', '?a ' + p[1] + ' ?b'),)] = len(d)
    #                                 filter_items[(('?a ' + p[0] + ' ?b', '?a ' + p[1] + ' ?b'),'?a ' + c + ' ?b')] = len(dd)
    #
    #         d = len({(b,a) for a,b in relations_ab[p[1]]}.intersection(relations_ab[p[0]]))
    #         if d >= miner.support:
    #             filter_items[('?b ' + p[0] + ' ?a', '?a ' + p[1] + ' ?b',)] = d


    ant_subs = {}
    ant_objs = {}

    _pr = list(itertools.product(cleaned_relations, repeat=2))

    cons_subs = {}
    cons_objs = {}
    #done_subs = set()
    #done_objs = set()
    #for p in tqdm(_pr):
    with Pool(mp.cpu_count() - 1, initializer=__init,
              initargs=(k_as_sub, k_as_obj, relations_ab,inv_relations_ab, miner.max_rule_set, miner.support, cleaned_single_rel)) as pool:

        for r in tqdm(pool.imap_unordered(exec_f1, _pr_comb, chunksize=1000), total=len(_pr_comb)):
            for el in r:
                filter_items[el] = r[el]

        for r in tqdm(pool.imap_unordered(exec, _pr, chunksize=1), total=len(_pr)):
            p, cons_sub, cons_objs, ant_subs, ant_objs = r
            print(p, ant_subs, ant_objs)
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



    # #
    # with Pool(6, initializer=__init, initargs=(k_as_sub, k_as_obj, relations_ab, miner.max_rule_set, miner.support)) as pool:
    #     for r in tqdm(pool.imap_unordered(__proc, _pr, chunksize=1000), total=len(_pr)):
    #         #t = (p,k_as_sub,k_as_obj,relations_ab,miner.max_rule_set)
    #         p, cons_sub_p, cons_objs_p, ant_subs_p0_p1_total ,ant_objs_p0_p1_total = r#__proc(t)
    #         #if (p[0],p[1]) not in ant_subs:
    #         #    ant_subs[(p[0],p[1])] = ant_subs_p0_p1
    #         #else:
    #         #    ant_subs[(p[0], p[1])] = ant_subs[(p[0],p[1])].union(ant_subs_p0_p1)
    #
    #         #if (p[0],p[1]) not in ant_objs:
    #         #    ant_objs[(p[0],p[1])] = ant_objs_p0_p1
    #         #else:
    #         #    ant_objs[(p[0], p[1])] = ant_objs[(p[0],p[1])].union(ant_objs_p0_p1)
    #         if cons_sub_p > 0:
    #             if p not in cons_subs:
    #                 cons_subs[p] = cons_sub_p
    #         if cons_objs_p > 0:
    #             if p not in cons_objs:
    #                 cons_objs[p] = cons_objs_p
    #
    #         if ant_subs_p0_p1_total > 0:
    #             if (p[0],p[1]) not in ant_subs:
    #                 ant_subs[(p[0],p[1])] = ant_subs_p0_p1_total
    #         if ant_objs_p0_p1_total > 0:
    #             if (p[0],p[1]) not in ant_objs:
    #                 ant_objs[(p[0],p[1])] = ant_objs_p0_p1_total
    #
    #
    #     # if p[0].count(':') + p[1].count(':') + p[2].count(':') <= miner.max_rule_set:
    #     #     ######################
    #     #     k1 = set(k_as_sub[p[0]].keys())
    #     #     k2 = set(k_as_sub[p[1]].keys())
    #     #     res = set()
    #     #     if (p[0],p[1],) not in ant_subs:
    #     #         ant_subs[(p[0],p[1],)] = set()
    #     #     for el in k2.intersection(k1):
    #     #         a = k_as_sub[p[0]][el]
    #     #         b = k_as_sub[p[1]][el]
    #     #         d = set([(x,y) for x in a for y in b])
    #     #         res.update(relations_ab[p[2]].intersection(d))
    #     #         if (p[0],p[1],) not in done_subs:
    #     #             ant_subs[(p[0],p[1],)].update(d)
    #     #     cons_subs[p] = len(res)
    #     #     if (p[0],p[1],) not in done_subs:
    #     #         ant_subs[(p[0], p[1],)] = len(ant_subs[(p[0],p[1],)])
    #     #         done_subs.add((p[0],p[1],))
    #     #     ##########################
    #     #     k1 = set(k_as_obj[p[0]].keys())
    #     #     k2 = set(k_as_obj[p[1]].keys())
    #     #     res2 = set()
    #     #     if (p[0],p[1],) not in ant_objs:
    #     #         ant_objs[(p[0],p[1],)] = set()
    #     #     for el in k2.intersection(k1):
    #     #         a = k_as_obj[p[0]][el]
    #     #         b = k_as_obj[p[1]][el]
    #     #         d = set([(x,y) for x in a for y in b])
    #     #         res2.update(relations_ab[p[2]].intersection(d))
    #     #         if (p[0], p[1],) not in done_objs:
    #     #             ant_objs[(p[0],p[1],)].update(d)
    #     #     cons_objs[p] = len(res2)
    #     #     if (p[0], p[1],) not in done_objs:
    #     #         ant_objs[(p[0], p[1],)] = len(ant_objs[(p[0],p[1],)])
    #     #         done_objs.add((p[0], p[1],))
    # for p in cons_subs:
    #     if cons_subs[p]>=miner.support:
    #         filter_items[(('?k '+p[0]+' ?a','?k '+p[1]+' ?b'),)]= ant_subs[(p[0],p[1],)]
    #         filter_items[(('?k '+p[0]+' ?a','?k '+p[1]+' ?b'),'?a '+p[2]+' ?b',)] = cons_subs[p]
    #
    # for p in cons_objs:
    #     if cons_objs[p]>=miner.support:
    #         filter_items[(('?a '+p[0]+' ?k','?b '+p[1]+' ?k'),)]= ant_objs[(p[0],p[1],)]
    #         filter_items[(('?a '+p[0]+' ?k','?b '+p[1]+' ?k'),'?a '+p[2]+' ?b',)] = cons_objs[p]

    df = pd.DataFrame(list(filter_items.items()), columns=['itemsets', 'support'])
    rules = association_rules(df, metric="support", min_threshold=miner.support)
    miner.rules = rules
    #1miner.rules = None

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

import functools, itertools, operator
def exec(p):
    cons_sub = {}
    ant_subs = 0

    cons_objs = {}
    ant_objs = 0

    if p[0].count(':') + p[1].count(':') <= rule_len-1:

        if len(relations_ab[p[0]]) >= support and len(relations_ab[p[1]]) >= support:
            dd = set(k_as_sub[p[0]].keys()).intersection(set(k_as_sub[p[1]].keys()))
            if np.sum([len(k_as_sub[p[0]][d]) * len(k_as_sub[p[1]][d]) for d in dd]) >=support:
                d1 = {(x,y) for d in dd for x in k_as_sub[p[0]][d] for y in k_as_sub[p[1]][d]}
            else:
                d1 = set()

            dd = set(k_as_obj[p[0]].keys()).intersection(set(k_as_obj[p[1]].keys()))
            if np.sum([len(k_as_obj[p[0]][d]) * len(k_as_obj[p[1]][d]) for d in dd]) >= support:
                d2 = {(x,y) for d in dd for x in k_as_obj[p[0]][d] for y in k_as_obj[p[1]][d] }
            else:
                d2 = set()

        #if p[0] != p[1]:
        #    k1 = set(k_as_sub[p[0]].keys())
        #    k2 = set(k_as_sub[p[1]].keys())
        #    dd = k2.intersection(k1)
        #else:
        #    dd = set(k_as_sub[p[0]].keys())
        #d1 = {(x, y) for el in dd for x in k_as_sub[p[0]][el] for y in k_as_sub[p[1]][el]}
        ant_subs = len(d1)


        #if p[0] != p[1]:
        #    k1 = set(k_as_obj[p[0]].keys())
        #    k2 = set(k_as_obj[p[1]].keys())
        #    dd = k2.intersection(k1)
        #else:
        #    dd = set(k_as_obj[p[0]].keys())
        #d2 = {(x, y) for el in dd for x in k_as_obj[p[0]][el] for y in k_as_obj[p[1]][el]}
        ant_objs = len(d2)

        for ant in cleaned_relations:
            if ant_subs >= support:
                cons_sub[ant] = len(d1.intersection(relations_ab[ant]))
            if ant_objs >= support:
                cons_objs[ant] = len(d2.intersection(relations_ab[ant]))

    return p,cons_sub,cons_objs,ant_subs,ant_objs
def __proc(t):
    p = t
    ant_subs_p0_p1 = set()
    ant_subs_p0_p1_total = 0
    cons_sub_p = 0
    ant_objs_p0_p1 = set()
    ant_objs_p0_p1_total = 0
    cons_objs_p = 0
    if p[0].count(':') + p[1].count(':') + p[2].count(':') <= rule_len:
        ######################
        k1 = set(k_as_sub[p[0]].keys())
        k2 = set(k_as_sub[p[1]].keys())
        dd = k2.intersection(k1)
        if len(dd)>=support:
            res = set()
            for el in dd:
                a = k_as_sub[p[0]][el]
                b = k_as_sub[p[1]][el]
                d = set([(x, y) for x in a for y in b])
                res.update(relations_ab[p[2]].intersection(d))
                ant_subs_p0_p1.update(d)
            cons_sub_p = len(res)
            ant_subs_p0_p1_total = len(ant_subs_p0_p1)
        ##########################
        k1 = set(k_as_obj[p[0]].keys())
        k2 = set(k_as_obj[p[1]].keys())
        dd = k2.intersection(k1)
        if len(dd) >= support:
            res2 = set()
            for el in dd:
                a = k_as_obj[p[0]][el]
                b = k_as_obj[p[1]][el]
                d = set([(x, y) for x in a for y in b])
                res2.update(relations_ab[p[2]].intersection(d))
                ant_objs_p0_p1.update(d)
            cons_objs_p = len(res2)
            ant_objs_p0_p1_total = len(ant_objs_p0_p1)

    return p, cons_sub_p, cons_objs_p, ant_subs_p0_p1_total ,ant_objs_p0_p1_total