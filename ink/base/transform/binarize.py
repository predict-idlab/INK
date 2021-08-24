"""
binarize.py file.
Defines all functions to transform the neighborhood of nodes of interest into the binary INK representation.
"""

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from bidict import bidict
import gc

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'

def increment_dict(key, dct, counter):
    if key not in dct:
        dct[key] = counter
        counter = counter + 1
    return dct, counter

# check if all elemants in list are floats:
def check_floats(lst):
    try:
        [float(l) for l in lst]
        return True
    except:
        return False


def create_tups(dataset, verbose, float_rpr):
    lst = []
    for tup in tqdm(dataset, disable=not verbose):
        n_dct = {}
        #ids.append(tup[0])
        for key in tup[1]:
            if isinstance(tup[1][key], list):
                if len(tup[1][key]) > 0:
                    #feature_map, counter = increment_dict(key, feature_map, counter)
                    n_dct[key] = np.int8(1)
                    if float_rpr and check_floats(tup[1][key]):
                        for dat in tup[1][key]:
                            # feature_map, counter = increment_dict(str(key + '§' + str(dat)), feature_map, counter)
                            n_dct[key+'_real_data'] = np.float(dat)
                        # float_rpr[feature_map[key]] = np.int8(1)
                    else:
                        for dat in tup[1][key]:
                            #feature_map, counter = increment_dict(str(key + '§' + str(dat)), feature_map, counter)
                            n_dct[key + '§' + str(dat)] = np.int8(1)  #
                            # n_dct[feature_map[key+'§'+str(dat)]] = np.int8(1)
            else:
                #feature_map, counter = increment_dict(key, feature_map, counter)
                n_dct[key] = np.int8(tup[1][key])
                # n_dct[feature_map[key]] = np.int8(tup[1][key])
        yield n_dct
        #lst.append(n_dct)
    #yield lst


def create_representation(dataset, verbose=True, float_rpr=False):
    """
    Function to create the binary INK representation.
    Transforms the neighborhood dictionaries into a sparse boolean matrix.

    :param dataset: Full dictionary with neighborhoods of nodes of interest.
    :type dataset: dict
    :param verbose: Parameter to show tqdm tracker (default False).
    :type verbose: bool
    :return: Tuple with the sparse boolean feature matrix, the nodes of interest index, the column feature names
    """

    #lst = []
    #ids = []
    #feature_map = bidict()
    #counter = 0

    #lst.append(n_dct)for tup in tqdm(dataset, disable=not verbose):
    #lst = [create_tups(tup) for tup in tqdm(dataset, disable=not verbose)]
    ids = [tup[0] for tup in dataset]

    if float_rpr is True:
        vec = DictVectorizer(sparse=True, dtype=float)
    else:
        vec = DictVectorizer(sparse=True, dtype=bool)
    features = vec.fit_transform(create_tups(dataset, verbose, float_rpr))
    return features, ids, vec.feature_names_
