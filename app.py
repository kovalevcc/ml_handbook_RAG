import os
import json
import streamlit as st

from retrieval.doc_loader import parse_html_with_langchain
from retrieval.text_splitter import split_documents
from retrieval.vectorstore import create_vectorstore
from model.llm_setup import get_llm
from embeddings.bge_embeddings import BGEEmbeddings
from langchain_community.vectorstores import FAISS
from model.rag_chain import create_rag_chain

DATA_DIR = "./data"
INDEX_PATH = "./faiss"
FILENAME2URL_PATH = os.path.join(DATA_DIR, "filename2url.json")


@st.cache_data(show_spinner=False)
def load_filename2url(path: str) -> dict:
    """Загружает словарь filename->url из JSON"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_vectorstore(index_path: str, data_dir: str):
    """
    Возвращает загруженный либо заново созданный векторный индекс FAISS.
    Кэшируем результат, чтобы при повторном запуске не пересоздавать.
    """
    embeddings = BGEEmbeddings()

    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        documents = parse_html_with_langchain(data_dir)
        split_docs = split_documents(documents)
        vectorstore = create_vectorstore(split_docs)
        vectorstore.save_local(index_path)

    return vectorstore


@st.cache_resource(show_spinner=False)
def get_llm_cached():
    """Возвращает LLM"""
    return get_llm()


def create_rag_chain_cached():
    """
    Создаёт RAG цепочку.
    """
    vectorstore = get_vectorstore(INDEX_PATH, DATA_DIR)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5})
    llm = get_llm_cached()
    return create_rag_chain(retriever, llm)


def process_query(question: str):
    """
    Принимает вопрос, вызывает RAG-цепочку и возвращает результат.
    """
    rag_chain = create_rag_chain_cached()
    response = rag_chain.invoke({"input": question})
    return response


# Основа страницы
st.set_page_config(
    page_title="RAGнарёк: Поисковик по ML учебнику Яндекса",
    page_icon="🛡️",
    layout="centered",
)


# Стили
def set_custom_style():
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f8ff;
            color: #021621;
        }

        .stForm {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
            padding: 0 !important;
        }

        .stButton, .stFormSubmitButton {
            display: flex;
            justify-content: center;
        }
        .stButton>button, .stFormSubmitButton>button {
            background-color: #021621;
            color: white;
            padding: 18px 60px;
            border: none;
            border-radius: 10px;
            margin-top: 8px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .stTextInput>div>div {
            border: none !important;
            padding: 0;
            box-shadow: none !important;
            background: transparent !important;
        }
        .stTextInput>div>div>input {
            border: 2px solid #003366;
            border-radius: 5px;
            text-align: center;
            display: flex;
            font-size: 18px;
            padding: 8px;
        }

        .vikings-theme {
            font-family: 'Georgia', serif;
            font-size: 24px;
            color: #021621;
            text-align: center;
            margin-bottom: 20px;
        }
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .title {
            font-size: 35px;
            text-align: center;
            color: #021621;
        }
        .description {
            background-color: #021621;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            align-items: center;
            flex-direction: column;
        }
        .descriptionfinal {
            background-color: #021621;
            color: white;
            padding: 20px;
            border-radius: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .links {
            color: #144272;
            padding: 20px;
            text-align: center;
            align-items: center;
            flex-direction: column;
        }
        .text-center {
            text-align: center;
            font-size: 22px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Фоновая картинка
def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://i.imgur.com/uHB4Rne.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True,
    )


add_bg_from_url()


# Сам сайт
def main():
    set_custom_style()

    st.markdown(
        """
        <div class="center">
            <img src="https://i.imgur.com/dLr3W5s.png" alt="RAG Logo" width="500">
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="title">🛡️ Добро пожаловать в RAGнарёк! 🛡️</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="text-center">Задавайте вопросы по учебнику Яндекса о машинном обучении, и мы найдем для вас ответы!</div>',
        unsafe_allow_html=True,
    )
    filename2url = load_filename2url(FILENAME2URL_PATH)
    # Ввод вопроса пользователем
    with st.form(key="question_form"):
        question = st.text_input(
            "Введите ваш вопрос:",
            placeholder="Например: Что такое lazy learning?"
        )
        submit_question = st.form_submit_button("Найти ответ")

    if submit_question:
        if question.strip():
            with st.spinner("Ищем ответ..."):
                # Запрашиваем ответ из RAG-цепочки
                answer = process_query(question)
                # Печатаем ответ
                st.success("Ответ найден!")
                st.write(answer.get('answer', "Не удалось получить ответ"))

                context_list = answer.get("context", [])

                if not context_list:
                    st.warning(
                        "Контекст не найден. Возможно, нет релевантных документов.")
                else:
                    # Берём первый документ
                    first_source = context_list[0].metadata.get("source", "")
                    # Находим URL
                    mapped_url = filename2url.get(
                        first_source, "URL not found")

                    st.write(f"Узнать больше: [по ссылке]({mapped_url})")
        else:
            st.warning("Пожалуйста, введите вопрос.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Описание проекта
    st.markdown(
        """
    <div class="description">
        <h4>❚❚❚ Что такое RAGнарёк? ❚❚❚</h4>
        <p>RAGнарёк — это интеллектуальная система, которая помогает находить ответы на вопросы, связанные с учебником Яндекса по машинному обучению (ML). Система RAGнарёк представляет собой оптимальное решение, которое сочетает в себе:</p>
        <ol>
            <li><b>Поиск информации в учебнике Яндекса по ML:</b>
            RAG-система извлекает релевантные данные из учебника.</li>
            <li><b>Генерацию понятных ответов:</b>
            На основе найденной информации система формирует текстовые ответы, которые легко воспринимаются пользователем.</li>
            <li><b>Удобный интерфейс:</b>
            Все это реализуется в интуитивно понятном и простом веб-приложении на Streamlit. </li>
        </ol>
        <h4>❚❚ Как это поможет пользователям? ❚❚</h4>
        <p>RAGнарёк — это ваш личный помощник для изучения машинного обучения. Он создан для того, чтобы сделать процесс изучения более доступным, понятным и увлекательным. RAGнарёк идеально подходит для:</p>
        <ul>
            <li>Студентов, изучающих машинное обучение.</li>
            <li>Специалистов, которым нужно быстро освежить знания.</li>
            <li>Всех, кто хочет получить доступ к информации из учебника Яндекса без необходимости читать весь текст.</li>
        </ul>
        <p><h4>❚ Пример использования ❚</h4>
        Вы хотите понять, как работает метод random forest. Вместо того, чтобы искать нужную главу в учебнике, вы просто задаёте вопрос системе, и она предоставляет вам краткий и точный ответ по заданной вами теме.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Ссылки
    st.markdown(
        """
        <div class="links">
            <h4>ᅠКонтактные данные</h4>
            <p>Telegram: <a href="https://t.me/@VadimPanenko" target="_blank">@RagNarek</a></p>
            <p>Email: <a href="mailto:ragnarek@mail.ru">ragnarek2002@mail.ru</a></p>
            <p><a href="https://education.yandex.ru/handbook/ml" target="_blank">Учебник Яндекса по ML</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Завершающий блок
    st.markdown(
        """
    <div class="descriptionfinal">
            <img src="https://i.imgur.com/DRl1Fy2.png" alt="RAG Logo" width="500">
            <p>Предоставьте функции поиска и генерации ответов нашему сервису, освобождая свое личное время для более глубокого понимания изучаемых материалов. <br> <br>RAGнарёк — ваш надежный источник информации и верный помощник на пути к новым открытиям в области машинного обучения!</p>
    </div>
                  </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
