import os
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.llms import VLLMOpenAI


def get_llm():
    llm = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="http://localhost:8000/v1",
        model_name="QWEN_AWQ",
        max_tokens=2000,
        model_kwargs={"stop": ["."]},
        temperature=0.1,
    )
    return llm


def get_llm():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    if not giga_key:
        raise ValueError(
            "Не установлен SB_AUTH_DATA в окружении. Установите ключ для GigaChat.")

    # Настраиваем GigaChat LLM
    llm = GigaChat(
        credentials=giga_key,
        model="GigaChat",    # Укажите нужную модель GigaChat
        timeout=30,
        verify_ssl_certs=False
    )
    return llm
