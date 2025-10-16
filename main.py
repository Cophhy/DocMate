try:
    import sys
    import pysqlite3  
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

from fastapi import FastAPI, Depends, HTTPException, Header, Body, Query, UploadFile, File
from pydantic import BaseModel
import ollama
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# PDF parsing
from pypdf import PdfReader
import shutil
from pathlib import Path
from uuid import uuid4

import chromadb

load_dotenv()

# Config and setting an API key for simple usage control
API_KEY = os.getenv("API_KEY", "dev-key")
API_KEY_CREDITS = {API_KEY: 100}

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")

CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral")         #"llama3" "mistral"
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

TOP_K_DEFAULT = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))               
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

PDF_DIR = Path(os.getenv("PDF_DIR", "./pdfs"))
PDF_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI()

@app.on_event("startup")
def _log_sqlite_version():
    import sqlite3
    print("SQLite em runtime:", sqlite3.sqlite_version)

def verify_api_key(x_api_key: str = Header(None)):
    credits = API_KEY_CREDITS.get(x_api_key, 0)
    if credits <= 0:
        raise HTTPException(status_code=401, detail="Invalid API Key, or no credits")
    return x_api_key

def get_chroma_client():
    from chromadb import PersistentClient  # Chroma >= 0.5
    return PersistentClient(path=CHROMA_PATH)

client = get_chroma_client()
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.strip().split())  
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for t in texts:
        r = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        embeddings.append(r["embedding"])
    return embeddings

def retrieve_context(question: str, k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    q_emb = embed_texts([question])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    return {
        "documents": res.get("documents", [[]])[0],
        "metadatas": res.get("metadatas", [[]])[0],
        "ids": res.get("ids", [[]])[0],
    }

def build_context_block(docs: List[str], metas: List[Dict[str, Any]]) -> str:
    parts = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        src = m.get("source", f"chunk-{i+1}")
        parts.append(f"[{i+1}] (source: {src})\n{d}")
    return "\n\n".join(parts)

def call_llm_with_context(prompt: str, context_block: str) -> str:
    system_inst = (
        "Você é um assistente RAG. Use APENAS o contexto fornecido para responder. "
        "Se a resposta não estiver no contexto, diga que não sabe. "
        "Seja direto e cite as fontes no final."
    )
    user_msg = f"Pergunta: {prompt}\n\n--- CONTEXTO ---\n{context_block}\n--- FIM CONTEXTO ---"
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_inst},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp["message"]["content"]

def safe_name(name: str) -> str:
    base = os.path.basename(name)
    base = base.replace(" ", "_")
    return base

def read_pdf_pages_text(pdf_path: Path) -> List[str]:
    texts: List[str] = []
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        t = page.extract_text() or ""
        t = t.replace("\x00", " ")
        texts.append(t)
    return texts

class IngestItem(BaseModel):
    text: str
    source: str | None = None
    id: str | None = None

class IngestBatch(BaseModel):
    items: List[IngestItem]

class GenerateBody(BaseModel):
    prompt: str
    top_k: int | None = None

@app.get("/health")
def health():
    return {"status": "ok", "models": {"chat": CHAT_MODEL, "embed": EMBED_MODEL}}

@app.get("/credits")
def credits(x_api_key: str = Depends(verify_api_key)):
    return {"api_key": x_api_key, "credits": API_KEY_CREDITS[x_api_key]}

@app.post("/ingest")
def ingest(batch: IngestBatch, x_api_key: str = Depends(verify_api_key)):
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for item in batch.items:
        src = item.source or "unspecified"
        chunks = chunk_text(item.text)
        for idx, ch in enumerate(chunks):
            doc_id = item.id or f"{src}-{uuid4()}-{idx}"
            ids.append(doc_id)
            docs.append(ch)
            metas.append({"source": src})

    if not docs:
        return {"ingested_chunks": 0, "unique_items": len(batch.items)}

    embs = embed_texts(docs)
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return {"ingested_chunks": len(docs), "unique_items": len(batch.items)}

@app.post("/reset_collection")
def reset_collection(x_api_key: str = Depends(verify_api_key)):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    global collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return {"status": "reset", "collection": COLLECTION_NAME}

@app.post("/generate")
def generate(
    body: GenerateBody | None = Body(None),
    prompt: str | None = Query(None),
    x_api_key: str = Depends(verify_api_key),
):
    user_prompt = body.prompt if body and body.prompt else prompt
    if not user_prompt:
        raise HTTPException(status_code=422, detail="Missing 'prompt'")

    API_KEY_CREDITS[x_api_key] -= 1

    k = body.top_k if (body and body.top_k is not None) else TOP_K_DEFAULT
    res = retrieve_context(user_prompt, k=k)
    docs = res["documents"]
    metas = res["metadatas"]

    if not docs:
        resp = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": user_prompt}])
        return {"response": resp["message"]["content"], "used_rag": False, "sources": []}

    context_block = build_context_block(docs, metas)
    answer = call_llm_with_context(user_prompt, context_block)

    sources = []
    for m in metas:
        src = m.get("source")
        if src and src not in sources:
            sources.append(src)

    return {"response": answer, "used_rag": True, "retrieved": len(docs), "sources": sources}


@app.get("/pdfs")
def list_pdfs(x_api_key: str = Depends(verify_api_key)):
    files = sorted([p.name for p in PDF_DIR.glob("*.pdf")])
    return {"count": len(files), "files": files, "dir": str(PDF_DIR.resolve())}

@app.post("/upload_pdfs")
async def upload_pdfs(
    files: List[UploadFile] = File(..., description="Envie 1+ PDFs"),
    x_api_key: str = Depends(verify_api_key),
):
    saved = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Arquivo não é PDF: {f.filename}")
        dest = PDF_DIR / safe_name(f.filename)
        # grava sem carregar tudo na memória
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest.name)
    return {"saved": saved, "dir": str(PDF_DIR.resolve())}

@app.post("/ingest_pdfs")
def ingest_pdfs(x_api_key: str = Depends(verify_api_key)):
    pdf_paths = list(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        return {"ingested_chunks": 0, "files": 0, "message": "Nenhum PDF encontrado em " + str(PDF_DIR.resolve())}

    total_chunks = 0
    added_ids: List[str] = []
    added_docs: List[str] = []
    added_metas: List[Dict[str, Any]] = []

    for pdf in pdf_paths:
        pages_text = read_pdf_pages_text(pdf)
        for idx, page_text in enumerate(pages_text, start=1):
            chunks = chunk_text(page_text)
            for c_idx, ch in enumerate(chunks):
                if not ch:
                    continue
                doc_id = f"{pdf.name}-p{idx}-c{c_idx}-{uuid4()}"
                added_ids.append(doc_id)
                added_docs.append(ch)
                added_metas.append({"source": f"{pdf.name}:p{idx}"})
                total_chunks += 1

    if not added_docs:
        return {"ingested_chunks": 0, "files": len(pdf_paths), "message": "Textos vazios/ilegíveis"}

    embs = embed_texts(added_docs)
    collection.add(ids=added_ids, documents=added_docs, metadatas=added_metas, embeddings=embs)

    return {
        "ingested_chunks": total_chunks,
        "files": len(pdf_paths),
        "sources_added": sorted(list({m["source"] for m in added_metas})),
    }
