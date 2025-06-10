from datasets import load_dataset
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import pickle  # To persist FAISS index

# Load dataset
dataset = load_dataset("viber1/indian-law-dataset", split="train")
df = dataset.to_pandas()

# Rename columns
df = df.rename(columns={"instruction": "Instruction", "output": "Response"})

# Create LangChain documents
documents = [
    Document(page_content=row["Response"], metadata={"question": row["Instruction"]})
    for _, row in df.iterrows()
]

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector store
vector_db = FAISS.from_documents(documents, embedding=embedding_model)

# Save FAISS index to disk
faiss_index_path = "legal_qa_faiss_index"
vector_db.save_local(faiss_index_path)

print("✅ FAISS Vector DB created and saved.")
