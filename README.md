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
Файлы проекта находятся в главной ветке в следующем порядке.
- **app.py**: Главный файл.
- **static/**: Директория для статических файлов.
  - **image/kartinka.png**: Картинка.
  - **image/kartinka.png**: Картинка....

Также проект разделен на отдельные ветки.

[Разработка](https://github.com/kovalevcc/ml_handbook_RAG/tree/dev)
- **файл.py**: Название
- **файл.py**: Название

[Интерфейс](https://github.com/kovalevcc/ml_handbook_RAG/tree/app)
- **app.py**: Главный файл.
- **static/**: Директория для статических файлов.
  - **image/kartinka.png**: Картинка.
  - **image/kartinka.png**: Картинка....

<i>Структура в процессе написания...</i>
### Установка <i>(на данный момент)</i>
1. Создать среду с Python 3.10.12.
2. Установить зависимости из файла requirements.txt 
3. Установить переменную среды GIGACHAT_CREDENTIALS c ключем API от Gigachat.
4. Добавить данные (html файлы) в папку data.
5. Опционально: добавить готовую векторную базу Faiss в папку faiss, для быстрого запуска без создания эмбеддингов заново.

### Запуск
```
streamlit run ./app/ragnarek.py
```

Для остановки приложения используйте сочетание клавиш `Ctrl + C` в терминале.

## Наша команда
![RAGnarekTeam.gif](https://github.com/kovalevcc/ml_handbook_RAG/blob/main/RAGnarekTeam.gif)
<br>— <b>Артем Ковалев</b>, <i>Роль. Краткое описание решенных задач.</i>
<br>— <b>Оксана Соломенчук</b>, <i>Роль. Краткое описание решенных задач.</i>
<br>— <b>Вадим Паненко</b>, <i>Роль. Краткое описание решенных задач.</i>