import random
import uuid
from typing import Any, Iterable

import faiss
import numpy as np

from core.interface import (AbstractData, AbstractDataSet,
                            AbstractEmbeddingModel, AbstractVectorStorage,
                            vector)

# ---------------------# Index Implementation #---------------------#


class FlatIndex[D: AbstractData](AbstractVectorStorage[D]):
    '''A storage for image embeddings using FAISS.'''

    def __init__(self, data: list[tuple[D, list[float]]]):
        '''Initialize the image vector storage from the image embeddings.'''
        embeddings, data_items_ = [], []
        for data_item, embedding in data:
            embeddings.append(embedding)
            data_items_.append(data_item)

        self.embeddings_ = np.array(embeddings).astype('float32')
        self.data_items_ = data_items_

        self.restore_embeddings = []
        self.restore_data_items = []
        if len(self.embeddings_) != len(self.data_items_):
            raise ValueError(
                'The number of embeddings and data items are different.')

        self.index_ = faiss.IndexFlatL2(self.embeddings_.shape[1])
        self.index_.add(self.embeddings_)

    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]] | list[tuple[float, D, vector]]:
        query_embedding = np.array([query]).astype(
            'float32')  # Convert to 2D numpy array
        distances, indices = self.index_.search(query_embedding, k)

        # Retrieve the nearest embeddings and their corresponding data items
        results: list[tuple[float, D]] = []
        for i in range(k):
            idx: int = indices[0][i]
            distance: float = distances[0][i]
            data_item = self.data_items_[idx]
            if with_vector:
                results.append((distance, data_item, self.embeddings_[idx]))
            else:
                results.append((distance, data_item))

        return results

    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.data_items_.index(item)
            ret.append(self.embeddings_[idx])
        return ret

    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        for data, vector in sended_data:
            self.restore_data_items.append(data)
            self.restore_embeddings.append(vector)

    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.restore_data_items.index(item)
            ret.append(self.restore_embeddings[idx])
        return ret

    @staticmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the L2 distance between the query and target vectors.'''
        qarr = np.array([query]).astype('float32')
        tarr = np.array([target]).astype('float32')
        qptr = faiss.swig_ptr(qarr)
        tptr = faiss.swig_ptr(tarr)
        return faiss.fvec_L2sqr(qptr, tptr, len(query))

    @staticmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D], echo: bool = False) -> 'FlatIndex[D]':
        import os
        import pickle

        # Prepare the data pairs
        data_pairs: list[tuple[D, vector]] = []
        for i, data in enumerate(dataset):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            data_pairs.append((data, data_emb))
            if echo:
                print(f"Embedded {i+1}/{len(dataset)}...", end="\r")
        # Create the index
        index = FlatIndex(data_pairs)
        if echo:
            print(f"Finished Building FlatIndex with {len(dataset)} Objects")
        return index


class FlatIPIndex[D: AbstractData](AbstractVectorStorage[D]):
    '''A storage for image embeddings using FAISS.'''

    def __init__(self, data: list[tuple[D, list[float]]]):
        '''Initialize the image vector storage from the image embeddings.'''
        embeddings, data_items_ = [], []
        for data_item, embedding in data:
            embeddings.append(embedding)
            data_items_.append(data_item)
    
        self.embeddings_ = np.array(embeddings).astype('float32')
        faiss.normalize_L2(self.embeddings_)
        self.data_items_ = data_items_

        self.restore_embeddings = []
        self.restore_data_items = []
        if len(self.embeddings_) != len(self.data_items_):
            raise ValueError('The number of embeddings and data items are different.')

        self.index_ = faiss.IndexFlatIP(self.embeddings_.shape[1])
        self.index_.add(self.embeddings_)

    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]] | list[tuple[float, D, vector]]:
        query_embedding = np.array([query]).astype('float32')  # Convert to 2D numpy array
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index_.search(query_embedding, k)

        # Retrieve the nearest embeddings and their corresponding data items
        results: list[tuple[float, D]] = []
        for i in range(k):
            idx: int = indices[0][i]
            distance: float = distances[0][i]
            data_item = self.data_items_[idx]
            if with_vector:
                results.append((distance, data_item, self.embeddings_[idx]))
            else:
                results.append((distance, data_item))

        return results
    
    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.data_items_.index(item)
            ret.append(self.embeddings_[idx])
        return ret

    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        for data, vector in sended_data:
            self.restore_data_items.append(data)
            self.restore_embeddings.append(vector)
    
    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.restore_data_items.index(item)
            ret.append(self.restore_embeddings[idx])
        return ret
        
    @staticmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the L2 distance between the query and target vectors.'''
        qarr = np.array([query]).astype('float32')
        tarr = np.array([target]).astype('float32')
        faiss.normalize_L2(qarr)
        faiss.normalize_L2(tarr)
        qptr = faiss.swig_ptr(qarr)
        tptr = faiss.swig_ptr(tarr)
        return faiss.fvec_inner_product(qptr, tptr, len(query))

    @staticmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D], echo: bool = False) -> 'FlatIPIndex[D]':
        import os
        import pickle
        # Prepare the data pairs
        data_pairs: list[tuple[D, vector]] = []
        for i, data in enumerate(dataset):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            data_pairs.append((data, data_emb))
            if echo:
                print(f"Embedded {i+1}/{len(dataset)}...", end="\r")
        # Create the index
        index = FlatIPIndex(data_pairs)
        if echo:
            print(f"Finished Building FlatIPIndex with {len(dataset)} Objects")
        return index

class HNSWIndex[D: AbstractData](AbstractVectorStorage[D]):
    '''A storage for image embeddings using FAISS.'''

    def __init__(self, data: list[tuple[D, list[float]]]):
        '''Initialize the image vector storage from the image embeddings.'''
        embeddings, data_items_ = [], []
        for data_item, embedding in data:
            embeddings.append(embedding)
            data_items_.append(data_item)

        self.embeddings_ = np.array(embeddings).astype('float32')
        self.data_items_ = data_items_

        if len(self.embeddings_) != len(self.data_items_):
            raise ValueError(
                'The number of embeddings and data items are different.')
        print("start create HNSWIndex")
        self.index_ = faiss.IndexHNSWFlat(self.embeddings_.shape[1], 32)
        self.index_.add(self.embeddings_)
        self.index_.hnsw.efSearch = 64
        print("finish create")

    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]]:
        query_embedding = np.array([query]).astype(
            'float32')  # Convert to 2D numpy array
        distances, indices = self.index_.search(query_embedding, k)

        # Retrieve the nearest embeddings and their corresponding data items
        results: list[tuple[float, D]] = []
        for i in range(k):
            idx: int = indices[0][i]
            distance: float = distances[0][i]
            data_item = self.data_items_[idx]
            results.append((distance, data_item))

        return results

    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        return

    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        return []

    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.data_items_.index(item)
            ret.append(self.embeddings_[idx])
        return ret

    @staticmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the L2 distance between the query and target vectors.'''
        qarr = np.array([query]).astype('float32')
        tarr = np.array([target]).astype('float32')
        qptr = faiss.swig_ptr(qarr)
        tptr = faiss.swig_ptr(tarr)
        return faiss.fvec_L2sqr(qptr, tptr, len(query))

    @staticmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D], echo: bool = False) -> 'HNSWIndex[D]':
        import os
        import pickle

        # Prepare the data pairs
        data_pairs: list[tuple[D, vector]] = []
        for i, data in enumerate(dataset):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            data_pairs.append((data, data_emb))
            if echo:
                print(f"Embedded {i+1}/{len(dataset)}...", end="\r")
        # Create the index
        index = HNSWIndex(data_pairs)
        if echo:
            print(f"Finished Building HNSWIndex with {len(dataset)} Objects")
        return index


class IVFPQIndex[D: AbstractData](AbstractVectorStorage[D]):
    '''A storage for image embeddings using FAISS.'''

    def __init__(self, data: list[tuple[D, list[float]]]):
        '''Initialize the image vector storage from the image embeddings.'''
        embeddings, data_items_ = [], []
        for data_item, embedding in data:
            embeddings.append(embedding)
            data_items_.append(data_item)

        data_pairs = list(zip(data_items_, embeddings))
        random.shuffle(data_pairs)
        data_items_, embeddings = zip(*data_pairs)
        self.embeddings_ = np.array(embeddings).astype('float32')
        self.data_items_ = data_items_

        if len(self.embeddings_) != len(self.data_items_):
            raise ValueError(
                'The number of embeddings and data items are different.')
        print("start create IVFPQIndex")

        nlist = int(4 * len(self.embeddings_) ** 0.5)
        m = 96
        nbits = 8
        quantizer = faiss.IndexFlatL2(self.embeddings_.shape[1])
        self.index_ = faiss.IndexIVFPQ(
            quantizer, self.embeddings_.shape[1], nlist, m, nbits)

        self.index_.train(self.embeddings_[:int(
            len(self.embeddings_) ** 0.5)*256])
        self.index_.nprobe = 1500
        self.index_.add(self.embeddings_)
        print("finish create")

    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]]:
        query_embedding = np.array([query]).astype(
            'float32')  # Convert to 2D numpy array
        distances, indices = self.index_.search(query_embedding, k)

        # Retrieve the nearest embeddings and their corresponding data items
        results: list[tuple[float, D]] = []
        for i in range(k):
            idx: int = indices[0][i]
            distance: float = distances[0][i]
            data_item = self.data_items_[idx]
            results.append((distance, data_item))

        return results

    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        return

    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        return []

    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.data_items_.index(item)
            ret.append(self.embeddings_[idx])
        return ret

    @staticmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the L2 distance between the query and target vectors.'''
        qarr = np.array([query]).astype('float32')
        tarr = np.array([target]).astype('float32')
        qptr = faiss.swig_ptr(qarr)
        tptr = faiss.swig_ptr(tarr)
        return faiss.fvec_L2sqr(qptr, tptr, len(query))

    @staticmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D], echo: bool = False) -> 'IVFPQIndex[D]':
        import os
        import pickle

        # Prepare the data pairs
        data_pairs: list[tuple[D, vector]] = []
        for i, data in enumerate(dataset):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            data_pairs.append((data, data_emb))
            if echo:
                print(f"Embedded {i+1}/{len(dataset)}...", end="\r")
        # Create the index
        index = IVFPQIndex(data_pairs)
        if echo:
            print(f"Finished Building IVFPQIndex with {len(dataset)} Objects")
        return index


class IVFFlatIndex[D: AbstractData](AbstractVectorStorage[D]):
    '''A storage for image embeddings using FAISS.'''

    def __init__(self, data: list[tuple[D, list[float]]]):
        '''Initialize the image vector storage from the image embeddings.'''
        embeddings, data_items_ = [], []
        for data_item, embedding in data:
            embeddings.append(embedding)
            data_items_.append(data_item)

        data_pairs = list(zip(data_items_, embeddings))
        random.shuffle(data_pairs)
        data_items_, embeddings = zip(*data_pairs)
        self.embeddings_ = np.array(embeddings).astype('float32')
        self.data_items_ = data_items_

        if len(self.embeddings_) != len(self.data_items_):
            raise ValueError(
                'The number of embeddings and data items are different.')
        print("start create IVFIndex")

        nlist = int(4 * len(self.embeddings_) ** 0.5)
        quantizer = faiss.IndexFlatL2(self.embeddings_.shape[1])

        self.index_ = faiss.IndexIVFFlat(
            quantizer, self.embeddings_.shape[1], nlist, faiss.METRIC_L2)

        self.index_.train(self.embeddings_[:int(
            len(self.embeddings_) ** 0.5)*256])
        self.index_.nprobe = 128
        self.index_.add(self.embeddings_)
        print("finish create")

    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]]:
        query_embedding = np.array([query]).astype(
            'float32')  # Convert to 2D numpy array
        distances, indices = self.index_.search(query_embedding, k)

        # Retrieve the nearest embeddings and their corresponding data items
        results: list[tuple[float, D]] = []
        for i in range(k):
            idx: int = indices[0][i]
            distance: float = distances[0][i]
            data_item = self.data_items_[idx]
            results.append((distance, data_item))

        return results

    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        return

    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        return []

    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            idx = self.data_items_.index(item)
            ret.append(self.embeddings_[idx])
        return ret

    @staticmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the L2 distance between the query and target vectors.'''
        qarr = np.array([query]).astype('float32')
        tarr = np.array([target]).astype('float32')
        qptr = faiss.swig_ptr(qarr)
        tptr = faiss.swig_ptr(tarr)
        return faiss.fvec_L2sqr(qptr, tptr, len(query))

    @staticmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D], echo: bool = False) -> 'IVFFlatIndex[D]':
        import os
        import pickle

        # Prepare the data pairs
        data_pairs: list[tuple[D, vector]] = []
        for i, data in enumerate(dataset):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            data_pairs.append((data, data_emb))
            if echo:
                print(f"Embedded {i+1}/{len(dataset)}...", end="\r")
        # Create the index
        index = IVFFlatIndex(data_pairs)
        if echo:
            print(
                f"Finished Building IVFFlatIndex with {len(dataset)} Objects")
        return index


class MilvusIndex[D: AbstractData](AbstractVectorStorage[D]):
    '''A storage for embeddings using Milvus Lite (L2 only).'''

    DEFAULT_URI: str = "./milvus_lite.db"
    COLLECTION_PREFIX: str = "fed_vector_"

    def __init__(self, data: list[tuple[D, list[float]]]):
        try:
            from pymilvus import MilvusClient  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "pymilvus is required for MilvusIndex. "
                "Please install pymilvus[milvus-lite]."
            ) from e
        if len(data) == 0:
            raise ValueError('The data for MilvusIndex cannot be empty.')

        embeddings, data_items_ = [], []
        for data_item, embedding in data:
            embeddings.append(embedding)
            data_items_.append(data_item)

        self.embeddings_ = np.array(embeddings).astype('float32')
        self.data_items_ = data_items_

        if len(self.embeddings_) != len(self.data_items_):
            raise ValueError('The number of embeddings and data items are different.')

        self.db_uri_ = self.DEFAULT_URI
        self.collection_name_ = f"{self.COLLECTION_PREFIX}{uuid.uuid4().hex}"
        self.client_ = MilvusClient(self.db_uri_)

        dim = int(self.embeddings_.shape[1])
        self.client_.create_collection(
            collection_name=self.collection_name_,
            dimension=dim,
            primary_field_name="id",
            id_type="string",
            vector_field_name="vector",
            metric_type="L2",
            auto_id=False,
            max_length=512,
        )

        self.data_item_map_: dict[str, D] = {}
        self.embedding_map_: dict[str, vector] = {}
        insert_data = []
        for data_item, embedding in zip(self.data_items_, self.embeddings_):
            item_id = data_item.label()
            if item_id in self.data_item_map_:
                raise ValueError(f'Duplicate label found: {item_id}')
            emb_list = embedding.tolist()
            self.data_item_map_[item_id] = data_item
            self.embedding_map_[item_id] = emb_list
            insert_data.append({"id": item_id, "vector": emb_list})

        self.client_.insert(
            collection_name=self.collection_name_,
            data=insert_data,
        )

    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]] | list[tuple[float, D, vector]]:
        limit = min(k, len(self.data_items_))
        if limit <= 0:
            return []

        query_embedding = np.array(query).astype('float32').tolist()
        raw_results = self.client_.search(
            collection_name=self.collection_name_,
            data=[query_embedding],
            limit=limit,
            output_fields=["vector"],
            search_params={"metric_type": "L2", "params": {}},
        )

        hits = raw_results[0] if len(raw_results) > 0 else []
        if with_vector:
            results_with_vector: list[tuple[float, D, vector]] = []
            for hit in hits:
                item_id = str(hit["id"])
                distance = float(hit["distance"])
                data_item = self.data_item_map_[item_id]
                entity = hit.get("entity", {})
                vector_value = entity.get("vector", self.embedding_map_[item_id])
                results_with_vector.append((distance, data_item, vector_value))
            return results_with_vector

        results: list[tuple[float, D]] = []
        for hit in hits:
            item_id = str(hit["id"])
            distance = float(hit["distance"])
            data_item = self.data_item_map_[item_id]
            results.append((distance, data_item))

        return results

    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            ret.append(self.embedding_map_[item.label()])
        return ret

    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        return

    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        return []

    @staticmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the L2 distance between the query and target vectors.'''
        qarr = np.array(query).astype('float32')
        tarr = np.array(target).astype('float32')
        return float(np.sum((qarr - tarr) ** 2))

    @staticmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D], echo: bool = False) -> 'MilvusIndex[D]':
        import os
        import pickle

        data_pairs: list[tuple[D, vector]] = []
        for i, data in enumerate(dataset):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            data_pairs.append((data, data_emb))
            if echo:
                print(f"Embedded {i+1}/{len(dataset)}...", end="\r")

        index = MilvusIndex(data_pairs)
        if echo:
            print(f"Finished Building MilvusIndex with {len(dataset)} Objects")
        return index


class QdrantIndex[D: AbstractData](AbstractVectorStorage[D]):
    '''A storage for embeddings using Qdrant local mode (L2 only).'''

    DEFAULT_LOCATION: str = ":memory:"
    COLLECTION_PREFIX: str = "fed_qdrant_"

    def __init__(self, data: list[tuple[D, list[float]]]):
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-not-found]
            from qdrant_client.models import Distance, PointStruct, VectorParams  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required for QdrantIndex. "
                "Please install qdrant-client."
            ) from e

        if len(data) == 0:
            raise ValueError('The data for QdrantIndex cannot be empty.')

        embeddings, data_items_ = [], []
        for data_item, embedding in data:
            embeddings.append(embedding)
            data_items_.append(data_item)

        self.embeddings_ = np.array(embeddings).astype('float32')
        self.data_items_ = data_items_

        if len(self.embeddings_) != len(self.data_items_):
            raise ValueError('The number of embeddings and data items are different.')

        dim = int(self.embeddings_.shape[1])
        for emb in self.embeddings_:
            if len(emb) != dim:
                raise ValueError('Embedding dimensions are inconsistent.')

        self.location_ = self.DEFAULT_LOCATION
        self.collection_name_ = f"{self.COLLECTION_PREFIX}{uuid.uuid4().hex}"
        self.client_ = QdrantClient(location=self.location_)

        self.client_.create_collection(
            collection_name=self.collection_name_,
            vectors_config=VectorParams(size=dim, distance=Distance.EUCLID),
        )

        self.data_item_map_: dict[int, D] = {}
        self.embedding_map_: dict[int, vector] = {}
        self.embedding_label_map_: dict[str, vector] = {}
        seen_labels: set[str] = set()

        points = []
        for idx, (data_item, embedding) in enumerate(zip(self.data_items_, self.embeddings_)):
            label = data_item.label()
            if label in seen_labels:
                raise ValueError(f'Duplicate label found: {label}')
            seen_labels.add(label)

            emb_list = embedding.tolist()
            self.data_item_map_[idx] = data_item
            self.embedding_map_[idx] = emb_list
            self.embedding_label_map_[label] = emb_list
            points.append(PointStruct(id=idx, vector=emb_list))

        self.client_.upsert(
            collection_name=self.collection_name_,
            points=points,
        )

    def search(self, query: vector, k: int, with_vector: bool = False) -> list[tuple[float, D]] | list[tuple[float, D, vector]]:
        limit = min(k, len(self.data_items_))
        if limit <= 0:
            return []

        query_embedding = np.array(query).astype('float32').tolist()
        query_resp = self.client_.query_points(
            collection_name=self.collection_name_,
            query=query_embedding,
            limit=limit,
            with_vectors=with_vector,
        )
        search_result = query_resp.points if hasattr(query_resp, "points") else query_resp

        def _read_field(obj: Any, field: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(field, default)
            return getattr(obj, field, default)

        def _normalize_vector(v: Any, fallback: vector) -> vector:
            if v is None:
                return fallback
            if isinstance(v, list):
                return v
            if isinstance(v, np.ndarray):
                return v.astype('float32').tolist()
            if isinstance(v, dict):
                if len(v) == 0:
                    return fallback
                first_val = next(iter(v.values()))
                if isinstance(first_val, list):
                    return first_val
                if isinstance(first_val, np.ndarray):
                    return first_val.astype('float32').tolist()
            return fallback

        if with_vector:
            results_with_vector: list[tuple[float, D, vector]] = []
            for point in search_result:
                item_id = int(_read_field(point, "id"))
                distance = float(_read_field(point, "score"))
                data_item = self.data_item_map_[item_id]
                vector_value = _normalize_vector(
                    _read_field(point, "vector", None),
                    self.embedding_map_[item_id],
                )
                results_with_vector.append((distance, data_item, vector_value))
            return results_with_vector

        results: list[tuple[float, D]] = []
        for point in search_result:
            item_id = int(_read_field(point, "id"))
            distance = float(_read_field(point, "score"))
            data_item = self.data_item_map_[item_id]
            results.append((distance, data_item))

        return results

    def get_vectors(self, items: Iterable[D]) -> list[vector]:
        ret = []
        for item in items:
            ret.append(self.embedding_label_map_[item.label()])
        return ret

    def restore(self, sended_data: Iterable[tuple[D, vector]]):
        return

    def get_from_restore(self, items: Iterable[D]) -> list[vector]:
        return []

    @staticmethod
    def distance(query: vector, target: vector) -> float:
        '''Return the L2 distance between the query and target vectors.'''
        qarr = np.array(query).astype('float32')
        tarr = np.array(target).astype('float32')
        return float(np.sum((qarr - tarr) ** 2))

    @staticmethod
    def build_from_dataset(dataset: AbstractDataSet[D], model: AbstractEmbeddingModel[D], echo: bool = False) -> 'QdrantIndex[D]':
        import os
        import pickle

        data_pairs: list[tuple[D, vector]] = []
        for i, data in enumerate(dataset):
            embpkl_path = data.label() + f".{model.__class__.__name__}.emb"
            if os.path.exists(embpkl_path):
                data_emb = pickle.load(open(embpkl_path, "rb"))
            else:
                data_emb = model.embed([data])[0]
                pickle.dump(data_emb, open(embpkl_path, "wb"))
            data_pairs.append((data, data_emb))
            if echo:
                print(f"Embedded {i+1}/{len(dataset)}...", end="\r")

        index = QdrantIndex(data_pairs)
        if echo:
            print(f"Finished Building QdrantIndex with {len(dataset)} Objects")
        return index
