from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch


class BGEEmbeddings(Embeddings):
    """
    Класс для использования BAAI/bge-m3 как эмбедер
    """

    def __init__(self, model_name="BAAI/bge-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def embed_text(self, text: str) -> list:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().tolist()

    def embed_documents(self, texts: list) -> list:
        return [self.embed_text(text)[0] for text in texts]

    def embed_query(self, text: str) -> list:
        return self.embed_text(text)[0]
