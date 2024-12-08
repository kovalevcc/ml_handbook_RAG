from langchain_community.vectorstores import FAISS
from embeddings.bge_embeddings import BGEEmbeddings


def create_vectorstore(documents):
    embeddings = BGEEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def get_retriever(vectorstore, k=5):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
