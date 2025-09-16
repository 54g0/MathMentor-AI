from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from openai import embeddings
from qdrant_client.http import models
import os
load_dotenv()
class KB_setup:
    def __init__(self,vector_db_dir):
        self.vector_db_dir = vector_db_dir
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    def load_data(self):
        vector_store_path = os.path.join(self.vector_db_dir,"vector_store")
        self.vector_store = FAISS.load_local(vector_store_path, self.embeddings,allow_dangerous_deserialization=True)
        return self.vector_store
    def create_vector_store(self):
        dataset = load_dataset("gsm8k","main")["train"]
        docs = []
        i = 0
        for item in dataset:
            docs.append(f"Q: {item['question']}\nA: {item['answer']}")
            i += 1
            if i == 500:
                break
        self.vector_store = FAISS.from_texts(docs, self.embeddings)
        vector_store_path = os.path.join(self.vector_db_dir,"vector_store")
        self.vector_store.save_local(vector_store_path)
        return "Vector store created and saved to disk."

kb_setup = KB_setup(vector_db_dir="provide the path to your vector db directory")
kb_setup.create_vector_store()
Vector_store = kb_setup.load_data()