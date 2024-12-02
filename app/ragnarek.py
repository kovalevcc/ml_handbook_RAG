import streamlit as st
from langchain.chains import RetrievalQA

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
        .stButton {
            display: flex;
            justify-content: center;
        }
        .stButton>button {
            background-color: #021621;
            color: white;
            padding: 18px 60px;
            border: none;
            border-radius: 10px;
            margin-top: 8px;
            margin-right: 28px;
            display: flex;
            justify-content: center;
            align-items: center;            
        }
        .stTextInput>div>div>input {
            border: 2px solid #003366;
            border-radius: 5px;
            text-align: center;
            display: flex;
            font-size: 18px;    
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

    # Ввод вопроса пользователем
    question = st.text_input(
        "Введите ваш вопрос:", placeholder="Например: Что такое градиентный спуск?"
    )
    if st.button("Найти ответ"):
        if question.strip():
            with st.spinner("Ищем ответ..."):
                answer = process_query(question, retriever)
                st.success("Ответ найден!")
                st.write(answer)
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
