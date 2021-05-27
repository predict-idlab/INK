"""
levels.py file.
Defines all required functions to handle numerical data during the transformation to the binary INK representation.
"""
from tqdm import tqdm

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'


def check_float(lvl):
    """
    Function to check whether all values within a list contains only float values

    :param lvl: List of possible numerical values.
    :type lvl: str
    :return: True when all values of the given list are numerical.
    :rtype: bool
    """
    try:
        [float(x) for x in lvl]
    except ValueError:
        return False
    return True


def create_levels(dct, dct_t, verbose=True):
    """
    Function which create the level columns from the numerical data.
    By using this function, additional features comparing numerical data with each other are added in the node's
    neighborhoods. These are later transformed into a binary representation.

    :param dct: Neighborhood dictionary from which numerical levels are added.
    :type dct: dict
    :param dct_t: Neighborhood dictionary from previously fitted neighborhoods (dct and dct_t are equal during the
                  fit_transform function, but different when using the transform function).
    :type dct_t: dict
    :param verbose: Parameter to show tqdm tracker (default False).
    :type verbose: bool
    :return: Original neighborhood with additional level-based information.
    :rtype: dict
    """
    if verbose:
        print('## create levels')
    level_counts = {}
    black_list = set()
    for tup in tqdm(dct, disable=not verbose):
        for key in tup[1]:
            if check_float(tup[1][key]):
                if key not in level_counts:
                    level_counts[key] = set()
                level_counts[key].update(tup[1][key])
            else:
                black_list.add(key)

    n_dct = []
    for tup in tqdm(dct_t, disable=not verbose):
        n_levels = {}
        for key in tup[1]:
            if key in level_counts and key not in black_list:
                for lvl in level_counts[key]:
                    try:
                        n_levels[key + '<=' + str(lvl)] = any(float(x) <= float(lvl) for x in tup[1][key] if x != '')
                        n_levels[key + '>=' + str(lvl)] = any(float(x) >= float(lvl) for x in tup[1][key] if x != '')
                    except Exception as exp:
                        print(exp)
                        continue

        n_levels.update(tup[1])
        n_dct.append((tup[0], n_levels))
    return n_dct
