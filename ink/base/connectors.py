"""
connector.py file.
Defines all required functions to make a connection to a knowledge graph.
"""

import json
import stardog
from urllib import parse
from rdflib import Graph
from abc import ABC, abstractmethod
import xmltodict
import time

try:
    import faster_than_requests as ftr
    ftr.set_headers(headers=[("Accept", "application/sparql-results+json")])
    fast = True
except ImportError as e:
    import requests
    from requests.adapters import HTTPAdapter
    from requests import Session
    fast = False

__author__ = 'Bram Steenwinckel'
__copyright__ = 'Copyright 2020, INK'
__credits__ = ['Filip De Turck, Femke Ongenae']
__license__ = 'IMEC License'
__version__ = '0.1.0'
__maintainer__ = 'Bram Steenwinckel'
__email__ = 'bram.steenwinckel@ugent.be'


class AbstractConnector(ABC):
    """
    Abstract Connector class
    Can be used to implement new types of connectors.

    The only function which have to be implemented is the query(self, str) function.
    This function is used to query the neighborhood.
    Makes sure this function return a dictionary in the correct format.
    """
    @abstractmethod
    def query(self, q_str):
        """
        Abstract query function.
        :param q_str: Query string.
        :type q_str: str
        :rtype: dict
        """
        pass


class StardogConnector(AbstractConnector):
    """
    A Stardog connector

    This Stardog connector class sets up the connection to a Stardog database.

    :param conn_details: a dictionary containing all the connection details.
    :type conn_details: dict
    :param database: database of interest.
    :type database: str

    Example::
        details = {'endpoint': 'http://localhost:5820'}
        connector = StardogConnector(details, "example_database")
    """
    def __init__(self, conn_details, database, reason=False):
        self.details = conn_details
        self.host = conn_details['endpoint']
        self.db = database
        self.reason = reason
        #self.connection =

        if not fast:
            self.session = Session()
            adapter = HTTPAdapter(pool_connections=10000, pool_maxsize=10000)
            self.session.mount('http://', adapter)
        #else:
        #    ftr.set_headers(headers=[("Accept", "application/sparql-results+json")])

    def delete_db(self):
        """
        Function to delete the delete the database when it is available
        :return: None
        """
        try:
            with stardog.Admin(**self.details) as admin:
                admin.database(self.db).drop()
        except Exception as ex:
            print(ex)
            print("no database to drop")

    def upload_kg(self, filename):
        """
        Uploads the knowledge graph to the previously initialized database.
        :param filename: The filename of the knowledge graph.
        :type filename: str
        :return: None
        """
        with stardog.Admin(**self.details) as admin:
            try:
                admin.new_database(self.db)
                with stardog.Connection(self.db, **self.details) as conn:
                    conn.begin()
                    conn.add(stardog.content.File(filename))
                    conn.commit()
            except Exception as ex:
                print(ex)

    def close(self):
        self.connection.close()

    def query(self, q_str):
        """
        Execute a query on the initialized Stardog database
        :param q_str: Query string.
        :type q_str: str
        :return: Dictionary generated from the ['results']['bindings'] json.
        :rtype: dict
        """
        conn = stardog.Connection(self.db, **self.details)
        r = conn.select(q_str)
        conn.close()
        return r['results']['bindings']

    def old_query(self, q_str):
        """
        Execute a query on the initialized Stardog database
        :param q_str: Query string.
        :type q_str: str
        :return: Dictionary generated from the ['results']['bindings'] json.
        :rtype: dict
        """

        query = parse.quote(q_str)
        if fast:
            #a = time.time()
            r = ftr.get2str(self.host + '/' + self.db + '/query?query=' + query+'&reasoning='+str(self.reason))
            ftr.close_client()

            #print(time.time()-a)
        else:
            r = self.session.get(self.host + '/' + self.db + '/query?query=' + query+'&reasoning='+str(self.reason),
                                 headers={"Accept": "application/sparql-results+json"}).text
        #conn = stardog.Connection(self.db, **self.details)
        #r = conn.select(q_str)
        return json.loads(r)['results']['bindings']


class RDFLibConnector(AbstractConnector):
    """
    A RDFLib connector.

    This RDFLib connector class stores the knowledge graph directly within memory.

    :param filename: The filename of the knowledge graph.
    :type filename: str
    :param dataformat: Dataformat of the knowledge graph. Use XML for OWL files.
    :type dataformat: str

    Example::
        connector = RDFLibConnector('example.owl', 'xml')
    """
    def __init__(self, filename, dataformat):
        self.db_type = 'rdflib'
        self.g = Graph()
        self.g.parse(filename, format=dataformat)

    def query(self, q_str):
        """
        Execute a query through RDFLib
        :param q_str: Query string.
        :type q_str: str
        :return: Dictionary generated from the ['results']['bindings'] json.
        :rtype: dict
        """
        res = self.g.query(q_str)
        return json.loads(res.serialize(format="json"))['results']['bindings']