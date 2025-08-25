import os
import hashlib
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from groq import Groq

app = FastAPI()

# Setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./data")
collection = client.get_or_create_collection("docs")

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise Exception("Need GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

class Question(BaseModel):
    text: str

def get_text_from_pdf(file_data):
    reader = PdfReader(file_data)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text, size=1000):
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    
    if file.filename.endswith('.pdf'):
        text = get_text_from_pdf(content)
    else:
        text = content.decode('utf-8')
    
    chunks = split_text(text)
    embeddings = embedder.encode(chunks).tolist()
    
    file_id = hashlib.md5(content).hexdigest()
    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"file": file.filename} for _ in chunks]
    )
    
    return {"status": "uploaded", "chunks": len(chunks)}

@app.post("/ask")
def ask_question(question: Question):
    q_embedding = embedder.encode([question.text]).tolist()
    
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=3
    )
    
    if not results["documents"][0]:
        return {"answer": "No documents found"}
    
    context = "\n".join(results["documents"][0])
    
    prompt = f"Context: {context}\n\nQuestion: {question.text}\nAnswer:"
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    
    return {"answer": answer}

@app.get("/")
def home():
    return {"message": "RAG API running"}