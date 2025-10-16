# DocMate

> RAG local com FastAPI + Ollama — faça upload de PDFs, ingira no Chroma e pergunte via API ou Streamlit. 100% local, sem chaves pagas.

## Pré-requisitos

- Python **3.10+**
- [Ollama](https://ollama.com) instalado e rodando (`ollama serve`)
- Modelos baixados no Ollama:
  ```bash
  ollama pull mistral
  ollama pull nomic-embed-text
  ```
- (Opcional) `git`, `curl`

## Instalando o DocMate

**Linux/macOS:**
```bash
git clone https://github.com/Cophhy/DocMate
cd DocMate

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/Cophhy/DocMate
cd DocMate

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz:
```bash
API_KEY=dev-key
CHROMA_PATH=./chroma_db
COLLECTION_NAME=docs
PDF_DIR=./pdfs
OLLAMA_CHAT_MODEL=mistral
OLLAMA_EMBED_MODEL=nomic-embed-text
```

## Usando o DocMate

1) **Inicie o Ollama** em outro terminal:
```bash
ollama serve
```

2) **Rode o backend** (FastAPI):
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3) **Rode o frontend** (Streamlit):
```bash
# no mesmo venv
export API_KEY=dev-key
export FRONTEND_BASE_URL="http://127.0.0.1:8000"
streamlit run app.py
```
No Windows:
```powershell
$env:API_KEY="dev-key"
$env:FRONTEND_BASE_URL="http://127.0.0.1:8000"
streamlit run app.py
```

4) **Exemplos de uso rápido por API:**

- Health:
```bash
curl http://127.0.0.1:8000/health
```

- Upload de PDFs:
```bash
curl -X POST "http://127.0.0.1:8000/upload_pdfs"   -H "x-api-key: dev-key"   -F "files=@/caminho/arquivo1.pdf"   -F "files=@/caminho/arquivo2.pdf"
```

- Listar PDFs:
```bash
curl -H "x-api-key: dev-key" http://127.0.0.1:8000/pdfs
```

- Ingerir TODOS os PDFs:
```bash
curl -X POST -H "x-api-key: dev-key" http://127.0.0.1:8000/ingest_pdfs
```

- Perguntar com RAG:
```bash
curl -X POST "http://127.0.0.1:8000/generate?prompt=Resuma%20os%20PDFs"   -H "x-api-key: dev-key"
```

> Alternativamente, use a UI do Streamlit para operar tudo sem código.


**Observações**  
- Pastas ignoradas por padrão: `chroma_db/`, `pdfs/`, `__pycache__/`, `.venv/` (veja `.gitignore`).  
- Se receber erro de **SQLite** no Chroma 0.5, o projeto já inclui o *shim* em `main.py`. Remova variáveis legadas `CHROMA_DB_IMPL`/`PERSIST_DIRECTORY` do ambiente caso necessário.
