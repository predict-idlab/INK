"""
counts.py file.
Defines all required functions to handle categorical data during the transformation to the binary INK representation.
"""

from tqdm import tqdm

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'


def create_counts(dct, verbose=True):
    """
    Function to add counts of the categorical data inside the neighborhood of nodes of interest.
    By using this function, additional pairs will be added to count the number of occurrences of certain values within
    the nodes neighborhood. A simple example can be the the total number of occurrences of a specific relation within
    a certain node of interest.

    :param dct: Dictionary with nodes of neighborhoods.
    :type dct: dict
    :param verbose: Parameter to show tqdm tracker (default False).
    :type verbose: bool
    :return: Original neighborhood with additional count-based information.
    :rtype: dict
    """
    if verbose:
        print('## create counts')
    n_dct = []
    for tup in tqdm(dct, disable=not verbose):
        n_counts = {}
        for key in tup[1]:
            for cnt in set(tup[1][key]):
                if tup[1][key].count(cnt) > 1:
                    n_counts['count.'+str(key)+'.'+str(cnt)] = [tup[1][key].count(cnt)]
            if len(tup[1][key]) > 1:
                n_counts['count.'+str(key)] = [len(tup[1][key])]
        n = tup[1]
        n.update(n_counts)
        n_dct.append((tup[0], n))
    return n_dct
