"""
graph.py file.
Defines all required functions to extract the neighborhoods within a knowledge graph.
"""

from tqdm import tqdm
import multiprocessing as mp
from functools import lru_cache
from multiprocessing import Pool
import gc

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'


class KnowledgeGraph:
    """
    Knowledge graph class representation

    This graph builds and stores the internal knowledge graph representations.
    It stores and builds the neighborhoods of the nodes of interest through the initialized connector.

    :param connector: Connector instance.
    :type connector: :py:class:`ink.base.connectors.AbstractConnector`
    :param prefixes: Optional dictionary of prefixes which should be mapped.
    :type prefixes: list
    """
    def __init__(self, connector, prefixes=None):
        if prefixes is None:
            prefixes = []
        self.connector = connector
        self.ind_instances = {}
        self.predicates = set()
        self.total_parts = {}
        self.neighbour_parts = {}
        self.prefixes = prefixes

    @lru_cache(maxsize=10000000)
    def neighborhood_request(self, noi):
        """
        Function to make a neighborhood request of a certain instance.

        :param noi: URI of Node Of Interest (noi).
        :type noi: str
        :return: Dictionary with all values specified as in the connector string.
        :rtype: dict
        """
        try:
            if noi[0] == '<':
                noi = noi[1:]
            if noi[-1] == '>':
                noi = noi[:-1]

            q = 'SELECT ?p ?o ?dt WHERE { BIND( IRI("' + noi + '") AS ?s ) ?s ?p ?o. BIND (datatype(?o) AS ?dt) }'
            #q = 'SELECT ?p ?o WHERE { <'+noi+'> ?p ?o. }'
            return self.connector.query(q)
        except Exception as e:
            print(e)
            return []

    def extract_neighborhoods(self, data, depth, skip_list=None, verbose=False, jobs=1):
        """
        Function which extracts all the neighborhoods of a given set of nodes of interest.

        :param data: List of node of interests.
        :type data: list
        :param depth: The maximum depth of the neighborhood.
        :type depth: int
        :param skip_list: List with relations which are not taken into account when traversing the neighborhoods.
        :type skip_list: list
        :param verbose: Parameter to show tqdm tracker (default False).
        :type verbose: bool
        :return: List of dictionaries containing the neighborhood until the defined depth for each instance of the
                 dataset.
        :rtype: list
        """

        if skip_list is None:
            skip_list = []

        seq =[(r, depth, skip_list) for r in data]
        if jobs > 1:
            with Pool(jobs) as pool:
                res = list(tqdm(pool.imap_unordered(self._create_neighbour_paths, seq, chunksize=10),
                                total=len(data), disable=not verbose))
                pool.close()
                pool.join()
        else:
            res = []
            for s in tqdm(seq, disable=not verbose, total=len(data)):
                res.append(self._create_neighbour_paths(s))
        return res

    def _create_neighbour_paths(self, t):
        """
        Internal function which capture the neighborhood of a single instance.

        :param t: Tuple containing the node of interest, the maximum depth and a list with relations to avoid.
        :type t: tuple
        :return: Tuple with, the node of interest and the corresponding neighborhood.
        :rtype: tuple
        """
        noi, depth, avoid_lst = t
        value = noi, ""
        total_parts = {}
        all_done = []
        res = self._define_neighborhood(value, depth, avoid_lst, total_parts, all_done)
        gc.collect()
        return noi, res

    def _replace_pref(self, r):
        """
        Internal function to strip the prefix from the given URI
        :param r: URI string
        :type r: str
        :return: URI string, with the prefix replaced.
        :rtype str
        """
        for x in self.prefixes:
            r = r.replace(x, self.prefixes[x])
            return r
        return r

    def _define_neighborhood(self, value, depth, avoid_lst, total_parts, all_done):
        """
        Internal function which defines the neighborhood of a node.
        This function is iterative, which means that it adds new nodes to be examined.

        :param value: Tuple containing the node to be examined and the edge from which it originated.
        :type value: tuple
        :param depth: depth of the current nodes.
        :type depth: int
        :param avoid_lst: List of relations and nodes to be avoided.
        :type avoid_lst: list
        :param total_parts: Dictionary used to keep track of already inserted relations.
                            This dictionary ensures multiple predicate, object relations can be stored when
                            the predicates are equal.
        :type total_parts: dict
        :param all_done: List of already processed nodes of interests, avoid cycles.
        :type all_done: list
        :return: The updated total_parts dictionary
        :rtype: dict
        """
        n_e, prop = value
        if depth > 0 and n_e not in all_done:
            res = self.neighborhood_request(n_e)
            next_noi = []
            for row in res:
                p = self._replace_pref(row['p']['value'])
                os = row['o']['value']
                if 'dt' in row:
                    dt = True
                else:
                    dt = False

                if not dt:
                    os = os.split(' ')
                else:
                    os = [os]

                for o in os:
                    if p not in avoid_lst and o not in avoid_lst:
                        if not dt:
                            if o.startswith('bnode'):
                                if prop == "":
                                    next_noi.append(('<_:' + o + '>', p))
                                else:
                                    next_noi.append(('<_:' + o + '>', prop + '.' + p))
                            else:
                                if prop == "":
                                    next_noi.append(('<' + o + '>', p))
                                    if p not in total_parts:
                                        total_parts[p] = list()
                                    total_parts[p].append(self._replace_pref(o))
                                else:
                                    next_noi.append(('<' + o + '>', prop + '.' + p))
                                    if prop + "." + p not in total_parts:
                                        total_parts[prop + "." + p] = list()
                                    total_parts[prop + "." + p].append(self._replace_pref(o))
                        else:
                            if prop == "":
                                if p not in total_parts:
                                    total_parts[p] = list()
                                total_parts[p].append(self._replace_pref(o))

                            else:
                                if prop + "." + p not in total_parts:
                                    total_parts[prop + "." + p] = list()
                                total_parts[prop + "." + p].append(self._replace_pref(o))
            if depth-1 > 0:
                [total_parts.update(self._define_neighborhood(value, depth - 1, avoid_lst, total_parts, all_done))
                 for value in next_noi]
            return total_parts
