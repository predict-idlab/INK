"""
binarize.py file.
Defines all functions to transform the neighborhood of nodes of interest into the binary INK representation.
"""

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'


def create_representation(dataset, verbose=True):
    """
    Function to create the binary INK representation.
    Transforms the neighborhood dictionaries into a sparse boolean matrix.

    :param dataset: Full dictionary with neighborhoods of nodes of interest.
    :type dataset: dict
    :param verbose: Parameter to show tqdm tracker (default False).
    :type verbose: bool
    :return: Tuple with the sparse boolean feature matrix, the nodes of interest index, the column feature names
    """
    lst = []
    ids = []
    for tup in tqdm(dataset, disable=not verbose):
        n_dct = {}
        ids.append(tup[0])
        for key in tup[1]:
            if isinstance(tup[1][key], list):
                if len(tup[1][key]) > 0:
                    n_dct[key] = np.int8(1)
                    for dat in tup[1][key]:
                        n_dct[key+'§'+str(dat)] = np.int8(1)
            else:
                n_dct[key] = np.int8(tup[1][key])
        lst.append(n_dct)

    vec = DictVectorizer(sparse=True, dtype=bool)
    features = vec.fit_transform(lst)
    return features, ids, vec.feature_names_
