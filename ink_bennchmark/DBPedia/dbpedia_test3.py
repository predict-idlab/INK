
import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
import numpy as np

import sys
import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from ink.base.connectors import StardogConnector
from ink.base.structure import InkExtractor
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.feature_selection import VarianceThreshold

from pyrdf2vec.graphs import KG
import pandas as pd
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from multiprocessing import Pool
from hashlib import md5
from typing import List,Set, Tuple, Any
from tqdm import tqdm
import rdflib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from pympler import asizeof

###
class MultiProcessingRandomWalker(RandomWalker):
    def _proc(self, t):
        kg, instance = t
        walks = self.extract_random_walks(kg, instance)
        canonical_walks = set()
        for walk in walks:
            canonical_walk = []
            for i, hop in enumerate(walk):  # type: ignore
                if i == 0 or i % 2 == 1:
                    canonical_walk.append(str(hop))
                else:
                    digest = md5(str(hop).encode()).digest()[:8]
                    canonical_walk.append(str(digest))
            canonical_walks.add(tuple(canonical_walk))

        return {instance:tuple(canonical_walks)}

    #overwrite this method
    def _extract(self, kg: KG, instances: List[rdflib.URIRef]) -> Set[Tuple[Any, ...]]:
        canonical_walks = set()
        seq = [(kg, r) for _,r in enumerate(instances)]
        #print(self.depth)
        with Pool(4) as pool:
            res = list(pool.imap_unordered(self._proc, seq))
        res = {k:v for element in res for k,v in element.items()}
        for r in instances:
            canonical_walks.update(res[r])

        return canonical_walks
###
import os.path
import csv

if not os.path.exists('results.csv'):
    with open('results.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Method', 'depth', 'NB_acc', 'NN_acc', 'DT_acc', 'SVC_acc', 'LR_acc', 'Extra_acc', 'Random_acc',
         'NB_weighted_F1', 'NN_weighted_F1', 'DT_weighted_F1', 'SVC_weighted_F1', 'LR_weighted_F1', 'Extra_weighted_F1', 'Random_weighted_F1',
         'NB_weighted_precision', 'NN_weighted_precision', 'DT_weighted_precision', 'SVC_weighted_precision', 'LR_weighted_precision', 'Extra_weighted_precision',
         'Random_weighted_precision', 'NB_weighted_recall', 'NN_weighted_recall', 'DT_weighted_recall', 'SVC_weighted_recall',
         'LR_weighted_recall', 'Extra_weighted_recall','Random_weighted_recall',
         'Create_time', 'NB_Train_time','NN_Train_time','DT_Train_time','SVC_Train_time','LR_Train_time','Extra_Train_time','Random_Train_time',
         'NB_Test_time', 'NN_Test_time', 'DT_Test_time', 'SVC_Test_time', 'LR_Test_time', 'Extra_Test_time','Random_Test_time',
         'Memory'])


""" parameters """
if __name__ == "__main__":

    prefs = {'http://www.w3.org/2005/Atom': 'a',
             'http://schemas.talis.com/2005/address/schema#': 'address',
             'http://webns.net/mvcb/': 'admin',
             'http://www.w3.org/ns/activitystreams#': 'as',
             'http://atomowl.org/ontologies/atomrdf#': 'atom',
             'http://soap.amazon.com/': 'aws',
             'http://b3s.openlinksw.com/': 'b3s',
             'http://schemas.google.com/gdata/batch': 'batch',
             'http://purl.org/ontology/bibo/': 'bibo',
             'bif:': 'bif',
             'http://www.openlinksw.com/schemas/bugzilla#': 'bugzilla',
             'http://www.w3.org/2002/12/cal/icaltzd#': 'c',
             'http://www.openlinksw.com/campsites/schema#': 'campsite',
             'http://www.crunchbase.com/': 'cb',
             'http://web.resource.org/cc/': 'cc',
             'http://purl.org/rss/1.0/modules/content/': 'content',
             'http://purl.org/captsolo/resume-rdf/0.2/cv#': 'cv',
             'http://purl.org/captsolo/resume-rdf/0.2/base#': 'cvbase',
             'http://www.w3.org/2001/sw/DataAccess/tests/test-dawg#': 'dawgt',
             'http://dbpedia.org/resource/Category:': 'dbc',
             'http://dbpedia.org/ontology/': 'dbo',
             'http://dbpedia.org/property/': 'dbp',
             'http://af.dbpedia.org/resource/': 'dbpedia-af',
             'http://als.dbpedia.org/resource/': 'dbpedia-als',
             'http://an.dbpedia.org/resource/': 'dbpedia-an',
             'http://ar.dbpedia.org/resource/': 'dbpedia-ar',
             'http://az.dbpedia.org/resource/': 'dbpedia-az',
             'http://bar.dbpedia.org/resource/': 'dbpedia-bar',
             'http://be.dbpedia.org/resource/': 'dbpedia-be',
             'http://be-x-old.dbpedia.org/resource/': 'dbpedia-be-x-old',
             'http://bg.dbpedia.org/resource/': 'dbpedia-bg',
             'http://br.dbpedia.org/resource/': 'dbpedia-br',
             'http://ca.dbpedia.org/resource/': 'dbpedia-ca',
             'http://commons.dbpedia.org/resource/': 'dbpedia-commons',
             'http://cs.dbpedia.org/resource/': 'dbpedia-cs',
             'http://cy.dbpedia.org/resource/': 'dbpedia-cy',
             'http://da.dbpedia.org/resource/': 'dbpedia-da',
             'http://de.dbpedia.org/resource/': 'dbpedia-de',
             'http://dsb.dbpedia.org/resource/': 'dbpedia-dsb',
             'http://el.dbpedia.org/resource/': 'dbpedia-el',
             'http://eo.dbpedia.org/resource/': 'dbpedia-eo',
             'http://es.dbpedia.org/resource/': 'dbpedia-es',
             'http://et.dbpedia.org/resource/': 'dbpedia-et',
             'http://eu.dbpedia.org/resource/': 'dbpedia-eu',
             'http://fa.dbpedia.org/resource/': 'dbpedia-fa',
             'http://fi.dbpedia.org/resource/': 'dbpedia-fi',
             'http://fr.dbpedia.org/resource/': 'dbpedia-fr',
             'http://frr.dbpedia.org/resource/': 'dbpedia-frr',
             'http://fy.dbpedia.org/resource/': 'dbpedia-fy',
             'http://ga.dbpedia.org/resource/': 'dbpedia-ga',
             'http://gd.dbpedia.org/resource/': 'dbpedia-gd',
             'http://gl.dbpedia.org/resource/': 'dbpedia-gl',
             'http://he.dbpedia.org/resource/': 'dbpedia-he',
             'http://hr.dbpedia.org/resource/': 'dbpedia-hr',
             'http://hsb.dbpedia.org/resource/': 'dbpedia-hsb',
             'http://hu.dbpedia.org/resource/': 'dbpedia-hu',
             'http://id.dbpedia.org/resource/': 'dbpedia-id',
             'http://ie.dbpedia.org/resource/': 'dbpedia-ie',
             'http://io.dbpedia.org/resource/': 'dbpedia-io',
             'http://is.dbpedia.org/resource/': 'dbpedia-is',
             'http://it.dbpedia.org/resource/': 'dbpedia-it',
             'http://ja.dbpedia.org/resource/': 'dbpedia-ja',
             'http://ka.dbpedia.org/resource/': 'dbpedia-ka',
             'http://kk.dbpedia.org/resource/': 'dbpedia-kk',
             'http://ko.dbpedia.org/resource/': 'dbpedia-ko',
             'http://ku.dbpedia.org/resource/': 'dbpedia-ku',
             'http://la.dbpedia.org/resource/': 'dbpedia-la',
             'http://lb.dbpedia.org/resource/': 'dbpedia-lb',
             'http://lmo.dbpedia.org/resource/': 'dbpedia-lmo',
             'http://lt.dbpedia.org/resource/as': 'dbpedia-lt',
             'http://lv.dbpedia.org/resource/a': 'dbpedia-lv',
             'http://mk.dbpedia.org/resource/': 'dbpedia-mk',
             'http://mr.dbpedia.org/resource/': 'dbpedia-mr',
             'http://ms.dbpedia.org/resource/': 'dbpedia-ms',
             'http://nah.dbpedia.org/resource/': 'dbpedia-nah',
             'http://nds.dbpedia.org/resource/': 'dbpedia-nds',
             'http://nl.dbpedia.org/resource/': 'dbpedia-nl',
             'http://nn.dbpedia.org/resource/': 'dbpedia-nn',
             'http://no.dbpedia.org/resource/': 'dbpedia-no',
             'http://nov.dbpedia.org/resource/': 'dbpedia-nov',
             'http://oc.dbpedia.org/resource/': 'dbpedia-oc',
             'http://os.dbpedia.org/resource/': 'dbpedia-os',
             'http://pam.dbpedia.org/resource/': 'dbpedia-pam',
             'http://pl.dbpedia.org/resource/': 'dbpedia-pl',
             'http://pms.dbpedia.org/resource/': 'dbpedia-pms',
             'http://pnb.dbpedia.org/resource/': 'dbpedia-pnb',
             'http://pt.dbpedia.org/resource/': 'dbpedia-pt',
             'http://ro.dbpedia.org/resource/': 'dbpedia-ro',
             'http://ru.dbpedia.org/resource/': 'dbpedia-ru',
             'http://sh.dbpedia.org/resource/': 'dbpedia-sh',
             'http://simple.dbpedia.org/resource/': 'dbpedia-simple',
             'http://sk.dbpedia.org/resource/': 'dbpedia-sk',
             'http://sl.dbpedia.org/resource/': 'dbpedia-sl',
             'http://sq.dbpedia.org/resource/': 'dbpedia-sq',
             'http://sr.dbpedia.org/resource/': 'dbpedia-sr',
             'http://sv.dbpedia.org/resource/': 'dbpedia-sv',
             'http://sw.dbpedia.org/resource/': 'dbpedia-sw',
             'http://th.dbpedia.org/resource/': 'dbpedia-th',
             'http://tr.dbpedia.org/resource/': 'dbpedia-tr',
             'http://ug.dbpedia.org/resource/': 'dbpedia-ug',
             'http://uk.dbpedia.org/resource/': 'dbpedia-uk',
             'http://vi.dbpedia.org/resource/': 'dbpedia-vi',
             'http://vo.dbpedia.org/resource/': 'dbpedia-vo',
             'http://war.dbpedia.org/resource/': 'dbpedia-war',
             'http://dbpedia.openlinksw.com/wikicompany/': 'dbpedia-wikicompany',
             'http://wikidata.dbpedia.org/resource/': 'dbpedia-wikidata',
             'http://yo.dbpedia.org/resource/': 'dbpedia-yo',
             'http://zh.dbpedia.org/resource/': 'dbpedia-zh',
             'http://zh-min-nan.dbpedia.org/resource/': 'dbpedia-zh-min-nan',
             'http://dbpedia.org/resource/': 'dbr',
             'http://dbpedia.org/resource/Template:': 'dbt',
             'http://purl.org/dc/elements/1.1/': 'dc',
             'http://purl.org/dc/terms/': 'dct',
             'http://digg.com/docs/diggrss/': 'digg',
             'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl': 'dul',
             'urn:ebay:apis:eBLBaseComponents': 'ebay',
             'http://purl.oclc.org/net/rss_2.0/enc#': 'enc',
             'http://www.w3.org/2003/12/exif/ns/': 'exif',
             'http://api.facebook.com/1.0/': 'fb',
             'http://api.friendfeed.com/2008/03': 'ff',
             'http://www.w3.org/2005/xpath-functions/#': 'fn',
             'http://xmlns.com/foaf/0.1/': 'foaf',
             'http://rdf.freebase.com/ns/': 'freebase',
             'http://base.google.com/ns/1.0': 'g',
             'http://www.openlinksw.com/schemas/google-base#': 'gb',
             'http://schemas.google.com/g/2005': 'gd',
             'http://www.w3.org/2003/01/geo/wgs84_pos#': 'geo',
             'http://sws.geonames.org/': 'geodata',
             'http://www.geonames.org/ontology#': 'geonames',
             'http://www.georss.org/georss/': 'georss',
             'http://www.opengis.net/gml': 'gml',
             'http://purl.org/obo/owl/GO#': 'go',
             'http://www.openlinksw.com/schemas/hlisting/': 'hlisting',
             'http://wwww.hoovers.com/': 'hoovers',
             'http://purl.org/stuff/hrev#': 'hrev',
             'http://www.w3.org/2002/12/cal/ical#': 'ical',
             'http://web-semantics.org/ns/image-regions': 'ir',
             'http://www.itunes.com/DTDs/Podcast-1.0.dtd': 'itunes',
             'http://www.w3.org/ns/ldp#': 'ldp',
             'http://linkedgeodata.org/triplify/': 'lgdt',
             'http://linkedgeodata.org/vocabulary#': 'lgv',
             'http://www.xbrl.org/2003/linkbase': 'link',
             'http://lod.openlinksw.com/': 'lod',
             'http://www.w3.org/2000/10/swap/math#': 'math',
             'http://search.yahoo.com/mrss/': 'media',
             'http://purl.org/commons/record/mesh/': 'mesh',
             'urn:oasis:names:tc:opendocument:xmlns:meta:1.0': 'meta',
             'http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#': 'mf',
             'http://musicbrainz.org/ns/mmd-1.0#': 'mmd',
             'http://purl.org/ontology/mo/': 'mo',
             'http://www.freebase.com/': 'mql',
             'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#': 'nci',
             'http://www.semanticdesktop.org/ontologies/nfo/#': 'nfo',
             'http://www.openlinksw.com/schemas/ning#': 'ng',
             'http://data.nytimes.com/': 'nyt',
             'http://www.openarchives.org/OAI/2.0/': 'oai',
             'http://www.openarchives.org/OAI/2.0/oai_dc/': 'oai_dc',
             'http://www.geneontology.org/formats/oboInOwl#': 'obo',
             'urn:oasis:names:tc:opendocument:xmlns:office:1.0': 'office',
             'http://www.opengis.net/': 'ogc',
             'http://www.opengis.net/ont/gml#': 'ogcgml',
             'http://www.opengis.net/ont/geosparql#': 'ogcgs',
             'http://www.opengis.net/def/function/geosparql/': 'ogcgsf',
             'http://www.opengis.net/def/rule/geosparql/': 'ogcgsr',
             'http://www.opengis.net/ont/sf#': 'ogcsf',
             'urn:oasis:names:tc:opendocument:xmlns:meta:1.0:': 'oo',
             'http://a9.com/-/spec/opensearchrss/1.0/': 'openSearch',
             'http://sw.opencyc.org/concept/': 'opencyc',
             'http://www.openlinksw.com/schema/attribution#': 'opl',
             'http://www.openlinksw.com/schemas/getsatisfaction/': 'opl-gs',
             'http://www.openlinksw.com/schemas/meetup/': 'opl-meetup',
             'http://www.openlinksw.com/schemas/xbrl/': 'opl-xbrl',
             'http://www.openlinksw.com/schemas/oplweb#': 'oplweb',
             'http://www.openarchives.org/ore/terms/': 'ore',
             'http://www.w3.org/2002/07/owl#': 'owl',
             'http://www.buy.com/rss/module/productV2/': 'product',
             'http://purl.org/science/protein/bysequence/': 'protseq',
             'http://www.w3.org/ns/prov#': 'prov',
             'http://backend.userland.com/rss2': 'r',
             'http://www.radiopop.co.uk/': 'radio',
             'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf',
             'http://www.w3.org/ns/rdfa#': 'rdfa',
             'http://www.openlinksw.com/virtrdf-data-formats#': 'rdfdf',
             'http://www.w3.org/2000/01/rdf-schema#': 'rdfs',
             'http://purl.org/stuff/rev#': 'rev',
             'http://purl.org/rss/1.0/': 'rss',
             'http://purl.org/science/owl/sciencecommons/': 'sc',
             'http://schema.org/': 'schema',
             'http://purl.org/NET/scovo#': 'scovo',
             'http://www.w3.org/ns/sparql-service-description#': 'sd',
             'urn:sobject.enterprise.soap.sforce.com': 'sf',
             'http://www.w3.org/ns/shacl#': 'sh',
             'http://www.w3.org/ns/shacl-shacl#': 'shsh',
             'http://rdfs.org/sioc/ns#': 'sioc',
             'http://rdfs.org/sioc/types#': 'sioct',
             'http://www.openlinksw.com/ski_resorts/schema#': 'skiresort',
             'http://www.w3.org/2004/02/skos/core#': 'skos',
             'http://purl.org/rss/1.0/modules/slash/': 'slash',
             'http://spinrdf.org/sp#': 'sp',
             'http://spinrdf.org/spin#': 'spin',
             'http://spinrdf.org/spl#': 'spl',
             'sql:': 'sql',
             'http://xbrlontology.com/ontology/finance/stock_market#': 'stock',
             'http://www.openlinksw.com/schemas/twfy#': 'twfy',
             'http://umbel.org/umbel#': 'umbel',
             'http://umbel.org/umbel/ac/': 'umbel-ac',
             'http://umbel.org/umbel/rc/': 'umbel-rc',
             'http://umbel.org/umbel/sc/': 'umbel-sc',
             'http://purl.uniprot.org/': 'uniprot',
             'http://dbpedia.org/units/': 'units',
             'http://www.rdfabout.com/rdf/schema/uscensus/details/100pct/': 'usc',
             'http://www.openlinksw.com/xsltext/': 'v',
             'http://www.w3.org/2001/vcard-rdf/3.0#': 'vcard',
             'http://www.w3.org/2006/vcard/ns#': 'vcard2006',
             'http://www.openlinksw.com/virtuoso/xslt/': 'vi',
             'http://www.openlinksw.com/virtuoso/xslt': 'virt',
             'http://www.openlinksw.com/schemas/virtcxml#': 'virtcxml',
             'http://www.openlinksw.com/schemas/virtpivot#': 'virtpivot',
             'http://www.openlinksw.com/schemas/virtrdf#': 'virtrdf',
             'http://rdfs.org/ns/void#': 'void',
             'http://www.worldbank.org/': 'wb',
             'http://www.w3.org/2007/05/powder-s#': 'wdrs',
             'http://www.w3.org/2005/01/wf/flow#': 'wf',
             'http://wellformedweb.org/CommentAPI/': 'wfw',
             'http://commons.wikimedia.org/wiki/': 'wiki-commons',
             'http://www.wikidata.org/entity/': 'wikidata',
             'http://en.wikipedia.org/wiki/': 'wikipedia-en',
             'http://www.w3.org/2004/07/xpath-functions': 'xf',
             'http://gmpg.org/xfn/11#': 'xfn',
             'http://www.w3.org/1999/xhtml': 'xhtml',
             'http://www.w3.org/1999/xhtml/vocab#': 'xhv',
             'http://www.xbrl.org/2003/instance': 'xi',
             'http://www.w3.org/XML/1998/namespace': 'xml',
             'http://www.ning.com/atom/1.0': 'xn',
             'http://www.w3.org/2001/XMLSchema#': 'xsd',
             'http://www.w3.org/XSL/Transform/1.0': 'xsl10',
             'http://www.w3.org/1999/XSL/Transform': 'xsl1999',
             'http://www.w3.org/TR/WD-xsl': 'xslwd',
             'urn:yahoo:maps': 'y',
             'http://dbpedia.org/class/yago/': 'yago',
             'http://yago-knowledge.org/resource/': 'yago-res',
             'http://gdata.youtube.com/schemas/2007': 'yt',
             'http://s.zemanta.com/ns#': 'zem'}

    rel = False

    dataset = sys.argv[1]#'BGS'#'BGS'
    depth = int(sys.argv[2])
    method = sys.argv[3]

    train = './' + dataset + '/TrainingSet.tsv'
    test = './' + dataset + '/TestSet.tsv'

    excludes = []#excludes_dict[dataset]#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']#['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://purl.org/collections/nl/am/objectCategory', 'http://purl.org/collections/nl/am/material']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://dl-learner.org/carcinogenesis#isMutagenic']#['http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis']#['http://swrc.ontoware.org/ontology#employs', 'http://swrc.ontoware.org/ontology#affiliation']

    labels_dict = {'AAUP': 'label_salary', 'albums': 'label', 'cities': 'label', 'forbes': 'label', 'movies': 'Label'}
    label_name = labels_dict[dataset]
    items_name = 'DBpedia_URL'

    #pos_file = 'mela/pos_mela.txt'
    #neg_file = 'mela/neg_mela.txt'

    df_train = pd.read_csv(train, delimiter='\t')
    df_train = df_train.dropna(subset=['label_name'])
    df_test = pd.read_csv(test, delimiter='\t')
    df_test = df_test.dropna(subset=['label_name'])

    data = pd.concat([df_train, df_test])

    le = preprocessing.LabelEncoder()
    df_train['label'] = le.fit_transform(df_train[label_name])
    df_test['label'] = le.transform(df_test[label_name])

    #print(df_train['label'].value_counts())
    #print(df_test['label'].value_counts())

    pos_file = set(['<' + x + '>' for x in data[items_name].values])

    ink_total_NB = []
    ink_total_NN = []
    ink_total_tree = []
    ink_total_support = []
    ink_total_log = []
    ink_total_extra = []
    ink_total_random = []

    ink_time_create = []
    ink_time_train = []
    ink_time_test = []

    ink_memory = []


    rdf_total_NB = []
    rdf_total_NN = []
    rdf_total_tree = []
    rdf_total_support = []
    rdf_total_log = []
    rdf_total_extra = []
    rdf_total_random = []

    rdf_time_create = []
    rdf_time_train = []
    rdf_time_test = []

    rdf_memory = []

    details = {'endpoint': 'http://10.2.35.70:5820'}
    connector = StardogConnector(details, "dbpedia")
    #connector.upload_kg(dir_kb+'/'+file)

    #for _ in tqdm(range(5)):

    ## INK exrtact

    if method=='INK':
        t0 = time.time()
        extractor = InkExtractor(connector, prefixes=prefs, verbose=True)
        X_train, _ = extractor.create_dataset(depth, pos_file, set(), excludes, jobs=4)
        extracted_data = extractor.fit_transform(X_train, counts=False, levels=False, float_rpr=True)

        df_data = pd.DataFrame.sparse.from_spmatrix(extracted_data[0])
        df_data.index = [x[1:-1] for x in extracted_data[1]]
        df_data.columns = extracted_data[2]

        print(df_data.shape)

        threshold_n = 0.9
        sel = VarianceThreshold(threshold_n * (1 - threshold_n))
        sel_var = sel.fit_transform(df_data)
        df_data = df_data[df_data.columns[sel.get_support(indices=True)]]

        print(df_data.shape)

        ink_time_create.append(time.time()-t0)

        ink_memory.append(asizeof.asizeof(df_data))


        df_train_extr = df_data[df_data.index.isin(df_train[items_name].values)]  # df_data.loc[[df_train['proxy']],:]
        df_test_extr = df_data[df_data.index.isin(df_test[items_name].values)]  # df_data.loc[[df_test['proxy']],:]

        df_train_extr = df_train_extr.merge(df_train[[items_name, 'label']], left_index=True, right_on=items_name)
        df_test_extr = df_test_extr.merge(df_test[[items_name, 'label']], left_index=True, right_on=items_name)

        ####
        X = df_train_extr.drop(['label', items_name], axis=1).values
        y = df_train_extr['label'].values

        clf_1 = KNeighborsClassifier(n_neighbors=3)
        clf_2 = GaussianNB()
        clf_3 = DecisionTreeClassifier()
        clf_4 = GridSearchCV(SVC(), {'C': [10 ** -3, 10 ** -2, 0.1, 1, 10, 10 ** 2, 10 ** 3]}, cv=3, n_jobs=4)
        clf_5 = GridSearchCV(LogisticRegression(),
                             {'C': [10 ** -3, 10 ** -2, 0.1, 1, 10, 10 ** 2, 10 ** 3], 'max_iter': [10000]}, cv=3,
                             n_jobs=4)
        clf_6 = ExtraTreesClassifier(n_estimators=100)
        clf_7 = RandomForestClassifier(n_estimators=100)

        # INK train:
        t1 = time.time()
        clf_1.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_2.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_3.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_4.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_5.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_6.fit(X, y)
        ink_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_7.fit(X, y)
        ink_time_train.append(time.time() - t1)


        # INK predict
        t2 = time.time()
        y_pred_1 = clf_1.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_2 = clf_2.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_3 = clf_3.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_4 = clf_4.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_5 = clf_5.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_6 = clf_6.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_7 = clf_7.predict(df_test_extr.drop(['label', items_name], axis=1).values)
        ink_time_test.append(time.time() - t2)

        print(y_pred_1)
        print(df_test_extr['label'].values)
        print(df_test_extr[items_name].values)


        ink_total_NN.append(accuracy_score(df_test_extr['label'].values, y_pred_1))
        ink_total_NN.append(f1_score(df_test_extr['label'].values, y_pred_1, average='weighted'))
        ink_total_NN.append(precision_score(df_test_extr['label'].values, y_pred_1, average='weighted'))
        ink_total_NN.append(recall_score(df_test_extr['label'].values, y_pred_1, average='weighted'))

        ink_total_NB.append(accuracy_score(df_test_extr['label'].values, y_pred_2))
        ink_total_NB.append(f1_score(df_test_extr['label'].values, y_pred_2, average='weighted'))
        ink_total_NB.append(precision_score(df_test_extr['label'].values, y_pred_2, average='weighted'))
        ink_total_NB.append(recall_score(df_test_extr['label'].values, y_pred_2, average='weighted'))

        ink_total_tree.append(accuracy_score(df_test_extr['label'].values, y_pred_3))
        ink_total_tree.append(f1_score(df_test_extr['label'].values, y_pred_3, average='weighted'))
        ink_total_tree.append(precision_score(df_test_extr['label'].values, y_pred_3, average='weighted'))
        ink_total_tree.append(recall_score(df_test_extr['label'].values, y_pred_3, average='weighted'))

        ink_total_support.append(accuracy_score(df_test_extr['label'].values, y_pred_4))
        ink_total_support.append(f1_score(df_test_extr['label'].values, y_pred_4, average='weighted'))
        ink_total_support.append(precision_score(df_test_extr['label'].values, y_pred_4, average='weighted'))
        ink_total_support.append(recall_score(df_test_extr['label'].values, y_pred_4, average='weighted'))

        ink_total_log.append(accuracy_score(df_test_extr['label'].values, y_pred_5))
        ink_total_log.append(f1_score(df_test_extr['label'].values, y_pred_5, average='weighted'))
        ink_total_log.append(precision_score(df_test_extr['label'].values, y_pred_5, average='weighted'))
        ink_total_log.append(recall_score(df_test_extr['label'].values, y_pred_5, average='weighted'))

        ink_total_extra.append(accuracy_score(df_test_extr['label'].values, y_pred_6))
        ink_total_extra.append(f1_score(df_test_extr['label'].values, y_pred_6, average='weighted'))
        ink_total_extra.append(precision_score(df_test_extr['label'].values, y_pred_6, average='weighted'))
        ink_total_extra.append(recall_score(df_test_extr['label'].values, y_pred_6, average='weighted'))

        ink_total_random.append(accuracy_score(df_test_extr['label'].values, y_pred_7))
        ink_total_random.append(f1_score(df_test_extr['label'].values, y_pred_7, average='weighted'))
        ink_total_random.append(precision_score(df_test_extr['label'].values, y_pred_7, average='weighted'))
        ink_total_random.append(recall_score(df_test_extr['label'].values, y_pred_7, average='weighted'))

        #Store
        with open('results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [dataset, 'INK', depth, ink_total_NB[0], ink_total_NN[0], ink_total_tree[0], ink_total_support[0],
                 ink_total_log[0], ink_total_extra[0], ink_total_random[0],ink_total_NB[1], ink_total_NN[1], ink_total_tree[1], ink_total_support[1],
                 ink_total_log[1], ink_total_extra[1], ink_total_random[1],ink_total_NB[2], ink_total_NN[2], ink_total_tree[2], ink_total_support[2],
                 ink_total_log[2], ink_total_extra[2], ink_total_random[2],ink_total_NB[3], ink_total_NN[3], ink_total_tree[3], ink_total_support[3],
                 ink_total_log[3], ink_total_extra[3], ink_total_random[3], ink_time_create[0], ink_time_train[0],ink_time_train[1],ink_time_train[2],ink_time_train[3],ink_time_train[4],ink_time_train[5], ink_time_train[6],
                 ink_time_test[0], ink_time_test[1], ink_time_test[2], ink_time_test[3], ink_time_test[4], ink_time_test[5], ink_time_test[6], ink_memory[0]])

    ## RDF2Vec extract:
    if method=='RDF2Vec':
        # extract
        t0 = time.time()
        kg = KG(location="http://localhost:5820/"+str(dataset)+"/query", is_remote=True, label_predicates=excludes)
        walkers = [RandomWalker(depth, 10000)]
        embedder = Word2Vec(size=500, sg=1)
        transformer = RDF2VecTransformer(walkers=walkers, embedder=embedder)
        inds = [ind[1:-1] for ind in list(pos_file)]
        embeddings = transformer.fit_transform(kg, inds)
        rdf_time_create.append(time.time()-t0)

        rdf_memory.append(asizeof.asizeof(embeddings))

        #Train
        t1 = time.time()
        train_inds = [inds.index(v) for v in df_train[items_name].values]
        test_inds = [inds.index(v) for v in df_test[items_name].values]

        X = [embeddings[i] for i in train_inds]
        y = df_train['label'].values

        clf_1 = KNeighborsClassifier(n_neighbors=3)
        clf_2 = GaussianNB()
        clf_3 = DecisionTreeClassifier()
        clf_4 = GridSearchCV(SVC(), {'C': [10 ** -3, 10 ** -2, 0.1, 1, 10, 10 ** 2, 10 ** 3]}, cv=3, n_jobs=4)
        clf_5 = GridSearchCV(LogisticRegression(),
                             {'C': [10 ** -3, 10 ** -2, 0.1, 1, 10, 10 ** 2, 10 ** 3], 'max_iter': [10000]}, cv=3,
                             n_jobs=4)
        clf_6 = ExtraTreesClassifier(n_estimators=100)
        clf_7 = RandomForestClassifier(n_estimators=100)

        # INK train:
        t1 = time.time()
        clf_1.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_2.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_3.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_4.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_5.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_6.fit(X, y)
        rdf_time_train.append(time.time() - t1)
        t1 = time.time()
        clf_7.fit(X, y)
        rdf_time_train.append(time.time() - t1)

        # predict

        t2 = time.time()
        y_pred_1 = clf_1.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_2 = clf_2.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_3 = clf_3.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_4 = clf_4.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_5 = clf_5.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_6 = clf_6.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)
        t2 = time.time()
        y_pred_7 = clf_7.predict([embeddings[i] for i in test_inds])
        rdf_time_test.append(time.time() - t2)

        rdf_total_NN.append(accuracy_score(df_test['label'].values, y_pred_1))
        rdf_total_NN.append(f1_score(df_test['label'].values, y_pred_1, average='weighted'))
        rdf_total_NN.append(precision_score(df_test['label'].values, y_pred_1, average='weighted'))
        rdf_total_NN.append(recall_score(df_test['label'].values, y_pred_1, average='weighted'))

        rdf_total_NB.append(accuracy_score(df_test['label'].values, y_pred_2))
        rdf_total_NB.append(f1_score(df_test['label'].values, y_pred_2, average='weighted'))
        rdf_total_NB.append(precision_score(df_test['label'].values, y_pred_2, average='weighted'))
        rdf_total_NB.append(recall_score(df_test['label'].values, y_pred_2, average='weighted'))

        rdf_total_tree.append(accuracy_score(df_test['label'].values, y_pred_3))
        rdf_total_tree.append(f1_score(df_test['label'].values, y_pred_3, average='weighted'))
        rdf_total_tree.append(precision_score(df_test['label'].values, y_pred_3, average='weighted'))
        rdf_total_tree.append(recall_score(df_test['label'].values, y_pred_3, average='weighted'))

        rdf_total_support.append(accuracy_score(df_test['label'].values, y_pred_4))
        rdf_total_support.append(f1_score(df_test['label'].values, y_pred_4, average='weighted'))
        rdf_total_support.append(precision_score(df_test['label'].values, y_pred_4, average='weighted'))
        rdf_total_support.append(recall_score(df_test['label'].values, y_pred_4, average='weighted'))

        rdf_total_log.append(accuracy_score(df_test['label'].values, y_pred_5))
        rdf_total_log.append(f1_score(df_test['label'].values, y_pred_5, average='weighted'))
        rdf_total_log.append(precision_score(df_test['label'].values, y_pred_5, average='weighted'))
        rdf_total_log.append(recall_score(df_test['label'].values, y_pred_5, average='weighted'))

        rdf_total_extra.append(accuracy_score(df_test['label'].values, y_pred_6))
        rdf_total_extra.append(f1_score(df_test['label'].values, y_pred_6, average='weighted'))
        rdf_total_extra.append(precision_score(df_test['label'].values, y_pred_6, average='weighted'))
        rdf_total_extra.append(recall_score(df_test['label'].values, y_pred_6, average='weighted'))

        rdf_total_random.append(accuracy_score(df_test['label'].values, y_pred_7))
        rdf_total_random.append(f1_score(df_test['label'].values, y_pred_7, average='weighted'))
        rdf_total_random.append(precision_score(df_test['label'].values, y_pred_7, average='weighted'))
        rdf_total_random.append(recall_score(df_test['label'].values, y_pred_7, average='weighted'))

        # store
        with open('results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
                [dataset, 'RDF2Vec', depth, rdf_total_NB[0], rdf_total_NN[0], rdf_total_tree[0], rdf_total_support[0],
                 rdf_total_log[0], rdf_total_extra[0], rdf_total_random[0], rdf_total_NB[1], rdf_total_NN[1],
                 rdf_total_tree[1], rdf_total_support[1],
                 rdf_total_log[1], rdf_total_extra[1], rdf_total_random[1], rdf_total_NB[2], rdf_total_NN[2],
                 rdf_total_tree[2], rdf_total_support[2],
                 rdf_total_log[2], rdf_total_extra[2], rdf_total_random[2], rdf_total_NB[3], rdf_total_NN[3],
                 rdf_total_tree[3], rdf_total_support[3],
                 rdf_total_log[3], rdf_total_extra[3], rdf_total_random[3], rdf_time_create[0], rdf_time_train[0],
                 rdf_time_train[1], rdf_time_train[2], rdf_time_train[3], rdf_time_train[4], rdf_time_train[5],
                 rdf_time_train[6],
                 rdf_time_test[0], rdf_time_test[1], rdf_time_test[2], rdf_time_test[3], rdf_time_test[4],
                 rdf_time_test[5], rdf_time_test[6], rdf_memory[0]])

    #
    #     #print(f'AUC LR: {accuracy_score(y_test, y_pred_1)}')
    # if ink_var:
    #     print('---INK----')
    #     print('INK Naive bayes')
    #     print(np.mean(ink_total_NB))
    #     print(np.std(ink_total_NB))
    #     print('INK Neirest neighbors')
    #     print(np.mean(ink_total_NN))
    #     print(np.std(ink_total_NN))
    #     print('INK Decision Tree')
    #     print(np.mean(ink_total_tree))
    #     print(np.std(ink_total_tree))
    #     print('INK SVC')
    #     print(np.mean(ink_total_support))
    #     print(np.std(ink_total_support))
    #     print('INK Logreg')
    #     print(np.mean(ink_total_log))
    #     print(np.std(ink_total_log))
    #     print('---')
    #     print('INK create time')
    #     print(np.mean(ink_time_create))
    #     print(np.std(ink_time_create))
    #     print('INK train time')
    #     print(np.mean(ink_time_train))
    #     print(np.std(ink_time_train))
    #     print('INK test time')
    #     print(np.mean(ink_time_test))
    #     print(np.std(ink_time_test))
    #     print('---')
    #     print('INK embedding size')
    #     print(np.mean(ink_memory))
    #     print(np.std(ink_memory))
    #
    # print('---RDF2Vec----')
    # print('RDF2Vec Naive bayes')
    # print(np.mean(rdf_total_NB))
    # print(np.std(rdf_total_NB))
    # print('RDF2Vec Neirest neighbors')
    # print(np.mean(rdf_total_NN))
    # print(np.std(rdf_total_NN))
    # print('RDF2Vec Decision Tree')
    # print(np.mean(rdf_total_tree))
    # print(np.std(rdf_total_tree))
    # print('RDF2Vec SVC')
    # print(np.mean(rdf_total_support))
    # print(np.std(rdf_total_support))
    # print('RDF2Vec Logreg')
    # print(np.mean(rdf_total_log))
    # print(np.std(rdf_total_log))
    # print('---')
    # print('RDF create time')
    # print(np.mean(rdf_time_create))
    # print(np.std(rdf_time_create))
    # print('RDF train time')
    # print(np.mean(rdf_time_train))
    # print(np.std(rdf_time_train))
    # print('RDF test time')
    # print(np.mean(rdf_time_test))
    # print(np.std(rdf_time_test))
    # print('---')
    # print('RDF embedding size')
    # print(np.mean(rdf_memory))
    # print(np.std(rdf_memory))

