import os

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from retrieval.doc_loader import parse_html_with_langchain
from retrieval.vectorstore import create_vectorstore
from model.llm_setup import get_llm
from embeddings.bge_embeddings import BGEEmbeddings
from langchain_community.vectorstores import FAISS
from model.rag_chain import create_rag_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Загрузка данных
file_path = 'data/TestDataset.csv'
data = pd.read_csv(file_path, sep=';')

# Проверяем наличие необходимых колонок
required_columns = ["query", "answer"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"В файле должны быть колонки: {required_columns}")

# Настройка векторного хранилища и модели
data_dir = "data"
index_path = "faiss"

embeddings = BGEEmbeddings()


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Разделение документов на чанки

    Args: 
        documents: Список документов
        chunk_size: Размер чанка
        chunk_overlap: Оверлап чанков

    Returns:
        split_docs: Список чанков
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# Функция для экспериментов с параметрами чанка и оверлапа


def experiment_with_chunk_params(documents, chunk_sizes, chunk_overlaps):
    """
    Эксперименты с различными параметрами чанков

    Args:
        documents: Список документов 
        chunk_sizes: Размеры чанков
        chunk_overlaps: Оверлапы чанков

    Returns:
        results_df: DataFrame с результатами
    """
    results = []
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            print(
                f"Параметры: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            split_docs = split_documents(documents, chunk_size, chunk_overlap)
            vectorstore = create_vectorstore(split_docs)

            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )
            llm = get_llm()
            rag_chain = create_rag_chain(retriever, llm)

            # Генерация предсказаний
            data["predicted_answer"] = data["query"].apply(
                lambda q: rag_chain.invoke({"input": q})
            )

            # Оценка метрик
            bleu_scores = []
            rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
            for _, row in data.iterrows():
                expected = str(row["answer"])
                predicted = str(row["predicted_answer"])

                # BLEU
                bleu_score = sentence_bleu(
                    [expected.split()], predicted.split())
                bleu_scores.append(bleu_score)

                # ROUGE
                scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True)
                rouge = scorer.score(expected, predicted)
                for key in rouge_scores.keys():
                    rouge_scores[key].append(rouge[key].fmeasure)

            bleu_avg = sum(bleu_scores) / len(bleu_scores)
            rouge_avg = {key: sum(values) / len(values)
                         for key, values in rouge_scores.items()}

            results.append({
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "bleu_avg": bleu_avg,
                **rouge_avg
            })

    return pd.DataFrame(results)


if os.path.exists(index_path):
    vectorstore = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
else:
    documents = parse_html_with_langchain(data_dir)
    split_docs = split_documents(documents)
    vectorstore = create_vectorstore(split_docs)
    vectorstore.save_local(index_path)

retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)
llm = get_llm()
rag_chain = create_rag_chain(retriever, llm)

# Проведение экспериментов с различными параметрами чанков
chunk_sizes = [500, 1000, 1500]
chunk_overlaps = [100, 200, 300]

print("Начало экспериментов...")
documents = parse_html_with_langchain(data_dir)
results_df = experiment_with_chunk_params(
    documents, chunk_sizes, chunk_overlaps)

# Сохранение результатов
results_df.to_csv("data/ChunkExperimentResults_2.csv", index=False)
print("Результаты экспериментов сохранены в ChunkExperimentResults_2.csv")
