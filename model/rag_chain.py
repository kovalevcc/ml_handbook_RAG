from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate


def create_rag_chain(retriever, llm):
    """
    Создание цепочки для модели RAG

    Args:
        retriever (Retriever): Экземпляр класса Retriever
        llm: Экземпляр модели

    Returns:    
        rag_chain: Цепочка для модели RAG
    """
    prompt = ChatPromptTemplate.from_messages([
        ("human", """"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Use only Russian language.
        Question: {input} 
        Context: {context} 
        Answer:"""),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain
