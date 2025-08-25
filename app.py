import os
import io
import hashlib
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# Libraries required by the assessment
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from groq import Groq

# -----------------------------
# Configuration
# -----------------------------
GROQ_MODEL = os.environ.get("GROQ_MODEL", "mixtral-8x7b-32768")  # or "llama3-8b-8192"
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking settings approximate ~500 tokens per chunk with ~50 token overlap
# As a simple heuristic, assume ~4 chars per token -> target ~2000 chars, overlap ~200 chars
CHUNK_SIZE_CHARS = int(os.environ.get("CHUNK_SIZE_CHARS", "2000"))  # Fixed: was 500, should be ~2000 for ~500 tokens
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "200"))  # Fixed: was 50, should be ~200 for ~50 tokens

# Limits
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_QUESTION_CHARS = 2000
TOP_K_MIN = 1  # Changed from 3 to 1
TOP_K_MAX = 10  # Changed from 5 to 10

# -----------------------------
# Initialize services
# -----------------------------
app = FastAPI(title="RAG Q&A with Groq + ChromaDB", version="1.0.0")

# Basic CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vector DB (ChromaDB) - persistent local directory
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_store")
try:
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
    COLLECTION_NAME = "documents_collection"
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

# Embedding model
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load embedding model: {e}")

# Groq client - FIXED: Now properly uses environment variable
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("Missing GROQ_API_KEY environment variable. Please set it before starting the application.")

try:
    groq_client = Groq(api_key=groq_api_key)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Groq client: {e}")

# -----------------------------
# Utilities
# -----------------------------
def filehash(content: bytes) -> str:
    """Generate SHA256 hash of file content"""
    return hashlib.sha256(content).hexdigest()

def read_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        texts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():  # Only add non-empty pages
                    texts.append(f"[Page {page_num + 1}]\n{page_text}")
            except Exception as e:
                print(f"Warning: Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        full_text = "\n\n".join(texts)
        if not full_text.strip():
            raise ValueError("No readable text found in PDF")
        return full_text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

def read_txt(file_bytes: bytes) -> str:
    """Read text from TXT file"""
    try:
        # Try UTF-8 first, then fall back to other encodings
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        # If all encodings fail, use utf-8 with error handling
        return file_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read text file: {e}")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Simple sliding-window chunking by characters to approximate token limits"""
    if not text or not text.strip():
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        
        if chunk and len(chunk) > 10:  # Only keep chunks with meaningful content
            chunks.append(chunk)
        
        if end >= text_len:
            break
            
        # Move start position with overlap
        start = max(start + chunk_size - overlap, start + 1)
        
        # Prevent infinite loop
        if start >= text_len:
            break
    
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts"""
    if not texts:
        return []
    
    try:
        # Handle potential memory issues with large batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = embedder.encode(
                batch, 
                show_progress_bar=False, 
                convert_to_numpy=False,
                normalize_embeddings=True  # Normalize for better cosine similarity
            )
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings
    except Exception as e:
        raise Exception(f"Failed to generate embeddings: {e}")

def build_prompt(context_chunks: List[str], question: str) -> str:
    """Build the prompt for the LLM with context and question"""
    context = "\n\n---\n\n".join(context_chunks)
    
    # Truncate context if too long (rough token limit)
    max_context_chars = 8000  # Leave room for question and system prompt
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n\n[Context truncated...]"
    
    prompt = f"""Based on the following context, answer the question as accurately and completely as possible.
If you cannot find a complete answer in the context, say "I don't have enough information to fully answer this question" and provide whatever relevant information you can find.

Context:
{context}

Question: {question}

Answer:"""
    return prompt

def validate_top_k(k: Optional[int]) -> int:
    """Validate and normalize top_k parameter"""
    if k is None:
        return TOP_K_MIN
    return max(TOP_K_MIN, min(TOP_K_MAX, int(k)))

# -----------------------------
# Schemas
# -----------------------------
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class DocumentInfo(BaseModel):
    document: str
    total_chunks: int

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks: int
    file_hash: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    total_sources: int

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/", summary="Root endpoint")
def read_root():
    """Root endpoint with basic API information"""
    return {
        "message": "RAG Q&A API with Groq + ChromaDB",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload documents",
            "ask": "POST /ask - Ask questions",
            "documents": "GET /documents - List documents",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        }
    }

@app.get("/documents", response_model=List[DocumentInfo])
def list_documents() -> List[DocumentInfo]:
    """List all uploaded documents with their chunk counts"""
    try:
        count = collection.count()
        if count == 0:
            return []
        
        # Get all documents with their metadata
        result = collection.get(include=["metadatas"])
        
        # Count chunks per document
        doc_counts: Dict[str, int] = {}
        for metadata in result.get("metadatas", []):
            source = metadata.get("source", "unknown")
            doc_counts[source] = doc_counts.get(source, 0) + 1
        
        return [DocumentInfo(document=doc, total_chunks=count) for doc, count in doc_counts.items()]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (PDF or TXT)"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES / (1024*1024):.1f} MB."
        )

    filename = file.filename.strip()
    lower_filename = filename.lower()

    # Extract text based on file type
    if lower_filename.endswith(".pdf"):
        text = read_pdf(content)
    elif lower_filename.endswith(".txt"):
        text = read_txt(content)
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Only .txt and .pdf files are supported."
        )

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in the document.")

    # Chunk the text
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no valid chunks after processing.")

    # Generate embeddings
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {e}")

    # Store in ChromaDB
    fh = filehash(content)
    ids = [f"{fh}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i, "file_hash": fh} for i in range(len(chunks))]

    try:
        # Check if document already exists
        existing = collection.get(ids=ids)
        if existing.get("ids"):
            raise HTTPException(
                status_code=409, 
                detail="Document already exists in the database."
            )
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store document chunks: {e}")

    return UploadResponse(
        message="Document processed and indexed successfully.",
        filename=filename,
        chunks=len(chunks),
        file_hash=fh
    )

@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    """Ask a question about the uploaded documents"""
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=413, 
            detail=f"Question too long. Maximum length is {MAX_QUESTION_CHARS} characters."
        )

    top_k = validate_top_k(payload.top_k)

    # Check if we have any documents
    try:
        doc_count = collection.count()
        if doc_count == 0:
            return AskResponse(
                answer="No documents have been uploaded yet. Please upload some documents first.",
                sources=[],
                total_sources=0
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check document count: {e}")

    # Generate embedding for the question
    try:
        question_embedding = embed_texts([question])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed question: {e}")

    # Perform vector search
    try:
        search_results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

    # Extract results
    documents = search_results.get("documents", [[]])[0] if search_results.get("documents") else []
    metadatas = search_results.get("metadatas", [[]])[0] if search_results.get("metadatas") else []
    distances = search_results.get("distances", [[]])[0] if search_results.get("distances") else []

    if not documents:
        return AskResponse(
            answer="I don't have enough information to answer this question.",
            sources=[],
            total_sources=0
        )

    # Prepare sources with relevance scores
    sources = []
    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        # Convert distance to relevance score (1 - distance for cosine similarity)
        relevance_score = max(0.0, min(1.0, 1.0 - float(distance)))
        
        sources.append({
            "document": metadata.get("source", "unknown"),
            "chunk_index": metadata.get("chunk_index", -1),
            "relevance_score": round(relevance_score, 4),
            "preview": doc[:200] + "..." if len(doc) > 200 else doc
        })

    # Build prompt and query LLM
    prompt = build_prompt(documents, question)

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that answers questions based on provided context. Always be accurate and cite the context when possible."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Slightly higher for more natural responses
            max_tokens=1024,  # Increased for longer answers
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {e}")

    if not answer:
        answer = "I don't have enough information to answer this question."

    return AskResponse(
        answer=answer,
        sources=sources,
        total_sources=len(sources)
    )

@app.delete("/documents")
def clear_documents():
    """Clear all documents from the database"""
    global collection
    try:
        # Get collection count first
        count = collection.count()
        
        if count == 0:
            return {"message": "No documents to clear.", "cleared_count": 0}
        
        # Reset the collection (this will clear all data)
        client.delete_collection(COLLECTION_NAME)
        
        # Recreate the collection
        
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        
        return {"message": f"Successfully cleared {count} document chunks.", "cleared_count": count}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {e}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        collection.count()
        
        # Test embedding model
        test_embedding = embed_texts(["test"])
        
        return {
            "status": "healthy",
            "database": "connected",
            "embedding_model": "loaded",
            "groq_client": "initialized"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)