from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from src.config import settings


class VectorStore:
    """
    Thin wrapper around ChromaDB with OpenAI embeddings.
    Handles document upsert, similarity search, and collection management.
    """

    COLLECTION_NAME = "financial_docs"

    def __init__(self, api_key: str = None, persist_dir: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.persist_dir = persist_dir or settings.chroma_persist_dir

        self.client = chromadb.Client()  # In-memory for session use

        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name=settings.embedding_model,
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return

        self.collection.upsert(
            ids=[doc["id"] for doc in documents],
            documents=[doc["content"] for doc in documents],
            metadatas=[
                {
                    "source": doc["source"],
                    "chunk_id": doc["chunk_id"],
                    "total_chunks": doc["total_chunks"],
                }
                for doc in documents
            ],
        )

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or settings.top_k_retrieval

        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "chunk_id": results["metadatas"][0][i]["chunk_id"],
                    "score": 1 - results["distances"][0][i],  # cosine similarity
                }
            )

        return output

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.ef,
        )
