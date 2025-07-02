import numpy as np
from FlagEmbedding import BGEM3FlagModel

class BGEEmbedding:
    def __init__(self):
        self.model = BGEM3FlagModel('BAAI/bge-base-en', use_fp16=False)

    def embed_documents(self, texts):
        texts = ["passage: " + t for t in texts]
        output = self.model.encode(texts)
        dense_vecs = output["dense_vecs"]
        dense_vecs = dense_vecs / np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        return dense_vecs.tolist()

    def embed_query(self, text):
        output = self.model.encode(["query: " + text])
        dense_vec = output["dense_vecs"][0]
        dense_vec = dense_vec / np.linalg.norm(dense_vec)
        return dense_vec.tolist()