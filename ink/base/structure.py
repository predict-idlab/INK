"""
structure.py file.
Defines the functions and classes to construct the binary INK representation.
"""

import numpy as np
from ink.base.graph import KnowledgeGraph
from ink.base.transform.counts import create_counts
from ink.base.transform.levels import create_levels
from ink.base.transform.binarize import create_representation

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'


class InkExtractor:
    """
    The INK extractor.
    Constructs the binary representation from a given knowledge graph.

    :param connector: Connector instance.
    :type connector: :py:class:`ink.base.connectors.AbstractConnector`
    :param prefixes: Optional dictionary of prefixes which should be mapped.
    :type prefixes: list
    :param verbose: Parameter to show tqdm tracker (default False).
    :type verbose: bool
    """
    def __init__(self, connector, prefixes=None, verbose=False):
        if prefixes is None:
            prefixes = []
        self.connector = connector
        self.kg = KnowledgeGraph(connector, prefixes)
        self.levels = {}
        self.verbose = verbose
        self.train_data = None

    def create_dataset(self, depth=4, pos=None, neg=None, skip_list=None, jobs=1):
        """
        Function to create the neighborhood dataset.
        Based on the input parameters, this function creates the dictionary of neighborhoods.

        The pos parameter is required, the neg is only required when a task specific case is being solved.
        The pos and neg parameters can be either: sets or query strings.
        When a query string is given, the connector which is associated with the InkExtractor will be used to get all
        the nodes of interest.

        :param depth: The maximal depth of the neighborhood
        :type depth: int
        :param pos: Set or query string describing the positive set of nodes of interest.
        :type pos: set, str
        :param neg: Set or query string describing the negative set of nodes of interest (only for task specific cases).
        :type neg: set, str
        :param skip_list: List with relations which are not taken into account when traversing the neighborhoods.
        :type skip_list: list
        :return: The extracted neighborhoods for each node of interest and the corresponding labels (only for task
        specific cases).
        """

        if skip_list is None:
            skip_list = []

        def _acquire_set(val):
            v_set = set()
            if isinstance(val, str):
                res = self.connector.query(val)
                for s in res:
                    v_set.add(s['s']['value'])
            else:
                if val is not None and not isinstance(val, set):
                    with open(val) as file:
                        v_set = set(['<' + r.rstrip("\n") + '>' for r in file.readlines()])
                else:
                    if isinstance(val, set):
                        v_set = val
            return v_set

        pos_set = _acquire_set(pos)
        neg_set = _acquire_set(neg)

        if self.verbose:
            print("#Process: get neighbourhood")

        all_noi = list(pos_set.union(neg_set))
        noi_neighbours = self.kg.extract_neighborhoods(all_noi, depth, skip_list, verbose=self.verbose, jobs=jobs)
        # update order
        all_noi = [n[0] for n in noi_neighbours]

        a = []
        if len(pos_set) > 0 or len(neg_set) > 0:
            for v in all_noi:
                if v in pos_set:
                    a.append(1)
                else:
                    a.append(0)

        return noi_neighbours, np.array(a)

    def fit_transform(self, dct, counts=False, levels=False, float_rpr=False):
        """
        Fit_transform function which transforms the neighborhood of several nodes of interest into
        a binary representation.

        :param dct: Dictionary containing the neighborhoods of multiple nodes of interest. Can be acquired by using the
        create_dataset function.
        :type dct: dict
        :param counts: Boolean indication if the `ink.base.modules.counts.create_counts` function must be used.
        :type counts: bool
        :param levels: Boolean indication if the `ink.base.modules.levels.create_levels` function must be used.
        :return: tuple with sparse feature matrix, list of indices (node of interests), feature names.
        :rtype: tuple
        """
        if self.verbose:
            print('# Transform')
        if counts:
            dct = create_counts(dct, verbose=self.verbose)

        self.train_data = dct

        if levels:
            dct = create_levels(dct, dct, verbose=self.verbose)

        cat_df = create_representation(dct, float_rpr=float_rpr, verbose=self.verbose)

        return cat_df

    def transform(self, dct, counts=False, levels=False):
        """
        Transform function which transforms the neighborhood of several nodes of interest into
        a binary representation. The main difference with the fit_transform function is that this function uses
        the previously fitted dictionary to make the representation.

        :param dct: Dictionary containing the neighborhoods of multiple nodes of interest. Can be acquired by using the
        create_dataset function.
        :type dct: dict
        :param counts: Boolean indication if the `ink.base.modules.counts.create_counts` function must be used.
        :type counts: bool
        :param levels: Boolean indication if the `ink.base.modules.levels.create_levels` function must be used.
        :return: tuple with sparse feature matrix, list of indices (node of interests), feature names.
        :rtype: tuple
        """
        if self.verbose:
            print('# Transform')
        if counts:
            dct = create_counts(dct, verbose=self.verbose)

        if levels:
            dct = create_levels(self.train_data, dct, verbose=self.verbose)

        cat_df = create_representation(dct, verbose=self.verbose)

        return cat_df
