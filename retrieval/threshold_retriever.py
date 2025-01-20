from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain.schema import BaseRetriever

from typing import Any, List

class ThresholdRetriever(BaseRetriever):
    """
    Кастомный ретривер для поиска документов по схожести с пороговым значением.

    Атрибуты:
        vectorstore (Any): Хранилище векторов для поиска.
        embeddings (Any): Модель для получения векторного представления запросов.
        threshold (float): Минимальное значение схожести для отбора документов (по умолчанию 0.05).
        k (int): Количество возвращаемых топовых документов (по умолчанию 5).

    Методы:
        get_relevant_documents(query: str) -> List[Document]:
            Возвращает документы, соответствующие пороговому значению схожести.
    """

    vectorstore: Any = Field(...)
    embeddings: Any = Field(...)
    threshold: float = Field(default=0.05)
    k: int = Field(default=5)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        vectorstore: Any,
        embeddings: Any,
        threshold: float = 0.05,
        k: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        object.__setattr__(self, "vectorstore", vectorstore)
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "k", k)

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_emb = self.embeddings.embed_query(query)
        docs_and_scores = self.vectorstore.similarity_search_with_score_by_vector(
            query_emb, k=self.k
        )

        relevant_docs = []
        for doc, distance in docs_and_scores:
            similarity = 1 - distance
            if similarity >= self.threshold:
                relevant_docs.append(doc)

        return relevant_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        raise NotImplementedError

    def delete_documents(self, **kwargs) -> None:
        raise NotImplementedError
