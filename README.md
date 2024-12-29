# RAGнарёк
![RAGnarekFirstAppTest.gif](https://github.com/kovalevcc/ml_handbook_RAG/blob/main/RAGnarekFirstAppTest.gif)
<br>RAGнарёк — это интеллектуальная система, которая помогает находить ответы на вопросы, связанные с учебником Яндекса по машинному обучению (ML).
## Описание проекта
Данный проект создается в рамках курса <b>Введение в большие языковые модели (LLM)</b>. В контексте курса была выбрана предложенная тема проекта от преподавателей, под названием "Поисковик по учебнику Яндекса по ML". В ходе работ по выбранной теме была создана идея проекта "RAGнарёк". Проект реализован в виде удобного веб-приложения на streamlit с возможностью осуществить поиск по учебнику Яндекса по ML, узнать более подробную информацию о проекте и перейти по ссылкам в блоке с контактными данными. Данный проект представляет собой оптимальное решение, которое сочетает в себе:
            <li><b>Поиск информации в учебнике Яндекса по ML:</b>  
            RAG-система извлекает релевантные данные из учебника.</li>
            <li><b>Генерацию понятных ответов:</b>  
            На основе найденной информации система формирует текстовые ответы, которые легко воспринимаются пользователем.</li>
            <li><b>Удобный интерфейс:</b>  
            Все это реализуется в интуитивно понятном и простом веб-приложении на Streamlit. </li>
### Как это поможет пользователям?
RAGнарёк — это ваш личный помощник для изучения машинного обучения. Он создан для того, чтобы сделать процесс изучения более доступным, понятным и увлекательным. RAGнарёк идеально подходит для:
            <li>Студентов, изучающих машинное обучение.</li>
            <li>Специалистов, которым нужно быстро освежить знания.</li>
            <li>Всех, кто хочет получить доступ к информации из учебника Яндекса без необходимости читать весь текст.</li>
### Пример использования
Вы хотите понять, как работает метод random forest. Вместо того, чтобы искать нужную главу в учебнике, вы просто задаёте вопрос системе, и она предоставляет вам краткий и точный ответ по заданной вами теме.
## Структура проекта
Файлы проекта находятся в главной ветке в следующем порядке:

```
ml_handbook_RAG/
├── embeddings/                  # Директория для работы с векторными представлениями.
│   └── bge_embeddings.py          # Скрипт для создания эмбеддингов на основе модели.
│
├── images/                      # Директория с изображениями для приложения.
│   ├── DRl1Fy2.png                # Изображение для оформления финального блока приложения.
│   ├── dLr3W5s.png                # Логотип приложения.
│   └── uHB4Rne.png                # Фоновое изображение.
│
├── data/                        # Директория с данными.
│   └── filename2url.json          # JSON-файл с маппингом страниц в векторной бд и URL.
│
├── faiss/                       # Директория с векторным индексом.
│   ├── index.faiss                # Основной индекс FAISS.
│   └── index.pkl                  # Метаданные индекса.
│
├── model/                       # Директория с файлами, связанными с настройкой моделей и цепочек.
│   ├── llm_setup.py               # Скрипт для настройки языковой модели (LLM).
│   └── rag_chain.py               # Реализация цепочки Retrieval-Augmented Generation (RAG).
│
├── parser/                      # Директория с файлами, связанными с парсингом данных.
│   └── html_parser.py             # Скрипт для парсинга HTML-документов.
│
├── retrieval/                   # Директория для модулей, связанных с извлечением данных.
│   ├── doc_loader.py              # Скрипт для загрузки документов.
│   ├── text_splitter.py           # Скрипт для разбиения текста на части для обработки.
│   └── vectorstore.py             # Скрипт для работы с векторным хранилищем данных.
│
├── experiments/                 # Директория с информацией о проведенных экспериментах.
│   ├── rag_evaluation.ipynb       # Тетрадь с оценкой RAG.
│   ├── retrievier_hp_search.py    # Скрипт для подбора оптимальных параметров векторизации.
│   └── README.md                  # Описание экспериментов.
│
├── RAGnarekFirstAppTest.gif       # Изображение с анимацией, кратко демонстрирующее работу приложения.
├── RAGnarekTeam.gif               # Изображение с нашей командой.
├── README.md                      # Основная документация проекта.
├── app.py                         # Главный файл приложения.
└── requirements.txt               # Список зависимостей для установки.
```

Также проект разделен на обособленные ветки, в которых велась разработка по отдельным аспектам создания приложения, до объединения в главной ветке. Это ветки по [разработке](https://github.com/kovalevcc/ml_handbook_RAG/tree/dev) и [интерфейсу](https://github.com/kovalevcc/ml_handbook_RAG/tree/app).

### Установка
1. Создать среду с Python 3.10.12.
2. Установить зависимости из файла requirements.txt

**conda**
```
conda create -n handbook python=3.10.12 
pip install -r requirements.txt
```
 
**poetry**
```
poetry install 
```
4. Установить переменную среды GIGACHAT_CREDENTIALS c ключем API от Gigachat.
```
Содержание файла:
GIGACHAT_CREDENTIALS=*gigachat api key*
```
6. [Опционально] Добавить данные (html файлы) в папку data. Чтобы сгенерировать индекс заново, необходимо удалить папку faiss.

### Запуск
**conda**
```
streamlit run ./app/app.py
```
**poetry**
```
poetry run streamlit run app.py
```

Для остановки приложения используйте сочетание клавиш `Ctrl + C` в терминале.

## Наша команда
![RAGnarekTeam.gif](https://github.com/kovalevcc/ml_handbook_RAG/blob/main/RAGnarekTeam.gif)
<br>— <b>Артем Ковалев</b>, @kovalevcc
<br>— <b>Оксана Соломенчук</b>, @sverhmassivnaya
<br>— <b>Вадим Паненко</b>, @VadimPanenko
