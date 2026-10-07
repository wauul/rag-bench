"""Reusable, fingerprinted Chroma index. Incomplete batches are safe to upsert again."""

import gc

from backend import retrieval
from backend.models import Configuration
from backend.profiling import measure
from backend.provenance import run_documents


class SharedIndex:
    def __init__(self, store, run, config, checkpoint, documents=None):
        self.store, self.run, self.data = store, run, config
        self.documents = documents
        self.config, self.checkpoint = Configuration(**config), checkpoint
        self.model = self.collection = None
        self.reused = False
        self.labels = {"configuration_id": config["id"]}

    def prepare(self):
        import chromadb
        from chromadb.config import Settings

        self.checkpoint()
        client = chromadb.PersistentClient(
            path=str(self.store.root / "chroma"), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = client.get_or_create_collection(
            f"run-{self.run['id']}-cfg-{self.data['id']}", metadata={"hnsw:space": "cosine"}
        )
        metadata = self.collection.metadata or {}
        fingerprint = self.run["input_fingerprint"]
        if (
            metadata.get("fingerprint") == fingerprint
            and metadata.get("chunk_count") == self.collection.count()
        ):
            self.reused = True
            with measure("index_reuse", **self.labels, cache="persisted_index"):
                pass
            return
        if metadata.get("fingerprint") and metadata["fingerprint"] != fingerprint:
            raise ValueError("Index fingerprint mismatch")
        with measure("chunking", **self.labels):
            chunks = retrieval.chunk_documents(
                self.documents
                if self.documents is not None
                else run_documents(self.store, self.run),
                self.config,
            )
        self.load_model()
        with measure("indexing", **self.labels, chunks=len(chunks), cache="cold_or_incomplete"):
            for start in range(0, len(chunks), 64):
                self.checkpoint()
                batch = chunks[start : start + 64]
                self.collection.upsert(
                    ids=[str(i) for i in range(start, start + len(batch))],
                    documents=[c["text"] for c in batch],
                    embeddings=retrieval.embed(self.model, [c["text"] for c in batch]),
                    metadatas=[{k: v for k, v in c.items() if k != "text"} for c in batch],
                )
            # The manifest is committed only after every batch succeeds.
            self.collection.modify(
                metadata={"fingerprint": fingerprint, "chunk_count": len(chunks)}
            )

    def load_model(self):
        if self.model is None:
            with measure("embedding_model_load", **self.labels, model=self.config.embedding_model):
                self.model = retrieval.load_embedder(self.config.embedding_model)

    def retrieve(self, question):
        self.checkpoint()
        self.load_model()
        query = retrieval.embed(
            self.model, [question], query=True, name=self.config.embedding_model
        )
        result = self.collection.query(
            query_embeddings=query, n_results=min(self.config.candidate_k, self.collection.count())
        )
        return [
            {**meta, "text": text, "distance": float(distance), "chunk_id": cid}
            for meta, text, distance, cid in zip(
                result["metadatas"][0],
                result["documents"][0],
                result["distances"][0],
                result["ids"][0],
            )
        ]

    def close(self):
        self.model = None
        gc.collect()
