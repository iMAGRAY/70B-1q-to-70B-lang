import numpy as np

METRIC_INNER_PRODUCT = 0

class SimpleIndex:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype=np.float32)
    @property
    def ntotal(self):
        return len(self.vectors)
    @property
    def is_trained(self):
        return True
    def train(self, vecs):
        pass
    def add(self, vecs):
        if not isinstance(vecs, np.ndarray):
            vecs = np.array(vecs, dtype=np.float32)
        self.vectors = np.vstack([self.vectors, vecs])
    def search(self, vec, top_k):
        if self.vectors.size == 0:
            scores = np.zeros((1, top_k), dtype=np.float32)
            ids = -np.ones((1, top_k), dtype=np.int64)
            return scores, ids
        dots = np.dot(self.vectors, vec[0])
        idx = np.argsort(dots)[::-1][:top_k]
        scores = dots[idx].astype(np.float32).reshape(1, -1)
        ids = idx.astype(np.int64).reshape(1, -1)
        return scores, ids

def index_factory(dim, _factory, _metric=METRIC_INNER_PRODUCT):
    return SimpleIndex(dim)

def normalize_L2(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms

def write_index(index, path):
    with open(path, 'wb') as f:
        np.save(f, index.vectors)

def read_index(path):
    with open(path, 'rb') as f:
        vecs = np.load(f)
    idx = SimpleIndex(vecs.shape[1])
    idx.vectors = vecs
    return idx

def get_num_gpus():
    return 0

def StandardGpuResources():
    return None

def index_cpu_to_gpu(res, dev, index):
    return index
