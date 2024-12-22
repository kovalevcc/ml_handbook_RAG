import os
from langchain_community.chat_models.gigachat import GigaChat
from dotenv import load_dotenv


def get_llm():
    """
    Получение экземпляра модели GigaChat

    Returns:
        GigaChat: Экземпляр модели GigaChat
    """

    load_dotenv()  # загружаем переменные окружения из файла .env

    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    if not giga_key:

        raise ValueError(
            "Не установлен GIGACHAT_CREDENTIALS в окружении."
            "Убедитесь, что ключ задан в .env или установлен в окружении.")

    llm = GigaChat(
        credentials=giga_key,
        model="GigaChat",
        timeout=30,
        verify_ssl_certs=False,
        temperature=0
    )
    return llm
