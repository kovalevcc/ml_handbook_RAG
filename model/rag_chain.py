from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate


def create_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        "Ответь на вопрос ниже, используя следующий контекст и не добавляя ничего лишнего. Отвечай только на русском языке \n\n{context}\n\nВопрос: {input}\n\nОтвет:"
    )

    # Создаем цепочку, комбинирующую документы (stuff chain)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    # Создаем Retrieval цепочку, которая будет использовать retriever и combine_docs_chain
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain
