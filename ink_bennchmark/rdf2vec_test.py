from pyrdf2vec.graphs import KG
import pandas as pd
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec


# Entities should be a list of URIs that can be found in the Knowledge Graph

from multiprocessing import Pool
from hashlib import md5
from typing import List,Set, Tuple, Any
from tqdm import tqdm
import rdflib

class MultiProcessingRandomWalker(RandomWalker):

    def _proc(self, t):
        kg, instance = t
        walks = self.extract_random_walks(kg, instance)
        for walk in walks:
            canonical_walk = []
            for i, hop in enumerate(walk):  # type: ignore
                if i == 0 or i % 2 == 1:
                    canonical_walk.append(str(hop))
                else:
                    digest = md5(str(hop).encode()).digest()[:8]
                    canonical_walk.append(str(digest))

        return {instance:tuple(canonical_walk)}

    #overwrite this method
    def _extract(self, kg: KG, instances: List[rdflib.URIRef]) -> Set[Tuple[Any, ...]]:
        canonical_walks = set()
        seq = [(kg, r) for _,r in enumerate(instances)]
        with Pool(1) as pool:
            res = list(tqdm(pool.imap_unordered(self._proc, seq),
                            total=len(seq)))
        res = {k:v for element in res for k,v in element.items()}
        for r in instances:
            canonical_walks.add(res[r])

        return canonical_walks


if __name__ == '__main__':
    kg = KG(location="http://10.2.35.70:5820/dbpedia/query", is_remote=True)

    walkers = [MultiProcessingRandomWalker(4, 200, UniformSampler())]
    embedder = Word2Vec(size=200)
    transformer = RDF2VecTransformer(walkers=walkers, embedder=embedder)

    benches = ['cities']#,'AAUP','forbes','albums','movies']
    for b in benches:
        df = pd.read_csv(b+'/CompleteDataset.tsv', delimiter='\t')
        #df_train = pd.read_csv(b+'/TrainingSet.csv', delimiter='\t').dropna()
        #df_test = pd.read_csv(b+'/TestSet.csv', delimiter='\t').dropna()
        entities = df['DBpedia_URL'].values

        print(len(entities))

        embeddings = transformer.fit_transform(kg, entities)
        print(len(embeddings), len(embeddings[0]))