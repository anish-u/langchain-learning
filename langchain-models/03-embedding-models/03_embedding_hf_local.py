from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

text = "What is the capital of India"

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France",
]

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
result = embedding.embed_query(text)
print(str(result))

result = embedding.embed_documents(documents)
print(str(result))