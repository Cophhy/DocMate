# app.py
import os
import io
import json
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG + Ollama Frontend", layout="wide")

# Session state
if "BASE_URL" not in st.session_state:
    st.session_state.BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://127.0.0.1:8000")
if "API_KEY" not in st.session_state:
    st.session_state.API_KEY = os.getenv("API_KEY", "dev-key")
if "history" not in st.session_state:
    st.session_state.history = []  # list of {prompt, response, sources}


def _headers():
    return {"x-api-key": st.session_state.API_KEY}


def api_get(path: str, params: dict | None = None):
    url = f"{st.session_state.BASE_URL}{path}"
    r = requests.get(url, headers=_headers(), params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post_json(path: str, payload: dict | None = None, params: dict | None = None):
    url = f"{st.session_state.BASE_URL}{path}"
    r = requests.post(
        url,
        headers={**_headers(), "Content-Type": "application/json"},
        json=payload,
        params=params,
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def api_post_files(path: str, files: list[tuple]):
    """
    files: list of tuples like ('files', (filename, bytes, 'application/pdf'))
    """
    url = f"{st.session_state.BASE_URL}{path}"
    r = requests.post(url, headers=_headers(), files=files, timeout=600)
    r.raise_for_status()
    return r.json()


# Sidebar
with st.sidebar:
    st.subheader("API Connection")
    st.text_input("Base URL", key="BASE_URL", help="e.g., http://127.0.0.1:8000")
    st.text_input("API Key", key="API_KEY", type="password")
    cols = st.columns(2)
    with cols[0]:
        if st.button("Health"):
            try:
                data = api_get("/health")
                st.success(f"OK: {data}")
            except Exception as e:
                st.error(f"Health failed: {e}")
    with cols[1]:
        if st.button("Credits"):
            try:
                data = api_get("/credits")
                st.info(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as e:
                st.error(f"Credits failed: {e}")

st.title("RAG + Ollama (Streamlit)")

# Tabs
rag_tab, upload_tab, pdfs_tab, ingest_text_tab, admin_tab = st.tabs(
    [
        "RAG Chat",
        "Upload PDFs",
        "PDFs & Ingestion",
        "Add Text",
        "Admin",
    ]
)

# RAG Chat Tab
with rag_tab:
    st.subheader("Ask with RAG")
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_area("Question", placeholder="e.g., Summarize the PDFs about project X", height=120)
    with col2:
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=4)
        send = st.button("Send", use_container_width=True)

    if send and prompt.strip():
        try:
            with st.spinner("Generating answer..."):
                data = api_post_json("/generate", payload={"prompt": prompt, "top_k": int(top_k)})
            st.session_state.history.append({
                "prompt": prompt,
                "response": data.get("response", ""),
                "sources": data.get("sources", []),
                "used_rag": data.get("used_rag", False),
                "retrieved": data.get("retrieved", 0),
            })
        except Exception as e:
            st.error(f"Failed to generate: {e}")

    # History (latest first)
    if st.session_state.history:
        st.divider()
        st.caption("History")
        for i, item in enumerate(reversed(st.session_state.history), start=1):
            st.markdown(f"**Q{i}:** {item['prompt']}")
            st.write(item["response"])
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.caption(f"RAG: {'Yes' if item.get('used_rag') else 'No'}")
            with meta_cols[1]:
                st.caption(f"Chunks: {item.get('retrieved', 0)}")
            with meta_cols[2]:
                if item.get("sources"):
                    st.caption("Sources:")
                    st.write("\n".join([f"- {s}" for s in item["sources"]]))

# Upload PDFs Tab
with upload_tab:
    st.subheader("Upload PDFs to the server")
    uploads = st.file_uploader("Select PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Upload", disabled=not uploads):
        if not uploads:
            st.warning("No files selected.")
        else:
            try:
                files = []
                for uf in uploads:
                    b = uf.read()  # read once
                    files.append(("files", (uf.name, b, "application/pdf")))
                with st.spinner("Uploading files..."):
                    res = api_post_files("/upload_pdfs", files)
                st.success(f"Uploaded: {res.get('saved', [])}")
                st.caption(f"Directory: {res.get('dir')}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

# PDFs & Ingestion Tab
with pdfs_tab:
    st.subheader("Manage PDFs & Ingestion")
    cols = st.columns(3)
    with cols[0]:
        if st.button("List PDFs"):
            try:
                data = api_get("/pdfs")
                st.info(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as e:
                st.error(f"Failed to list PDFs: {e}")
    with cols[1]:
        if st.button("Ingest ALL PDFs"):
            try:
                with st.spinner("Ingesting PDFs into the vector store..."):
                    data = api_post_json("/ingest_pdfs", payload=None)
                st.success("Ingestion completed!")
                st.info(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as e:
                st.error(f"Failed to ingest PDFs: {e}")
    with cols[2]:
        if st.button("View Credits"):
            try:
                data = api_get("/credits")
                st.info(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as e:
                st.error(f"Failed: {e}")

# Add Text Tab
with ingest_text_tab:
    st.subheader("Ingest Text Manually")
    source = st.text_input("Source", placeholder="e.g., wiki-python")
    text = st.text_area("Text to ingest", height=180)
    if st.button("Ingest text"):
        if not text.strip():
            st.warning("Provide text to ingest")
        else:
            payload = {"items": [{"text": text, "source": source or "manual"}]}
            try:
                with st.spinner("Creating embeddings and saving..."):
                    data = api_post_json("/ingest", payload=payload)
                st.success("Ingestion completed!")
                st.info(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as e:
                st.error(f"Failed to ingest: {e}")

# Admin Tab
with admin_tab:
    st.subheader("Administration")
    if st.button("Reset vector collection"):
        try:
            with st.spinner("Resetting collection..."):
                data = api_post_json("/reset_collection", payload=None)
            st.success("Collection reset.")
            st.info(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            st.error(f"Failed to reset: {e}")

    st.caption("Tip: use the sidebar 'Health' and 'Credits' buttons to check status and consumption.")
