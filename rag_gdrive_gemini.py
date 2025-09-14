"""
RAG pipeline (simple, beginner-friendly):
- Poll a Google Drive folder (service-account) and download files
- Extract text from PDFs / DOCX / TXT
- Chunk text into overlapping chunks
- Create embeddings using Google GenAI (Gemini embedding)
- Upsert chunks to Pinecone
- Expose a simple Flask endpoint to chat (query -> retrieve -> ask Gemini chat)
"""

import os
import io
import time
import uuid
from dotenv import load_dotenv
from typing import List, Dict

# Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Google GenAI (Gemini) SDK (official)
from google import genai

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# Text extraction
from pypdf import PdfReader        # pip install pypdf
from docx import Document         # pip install python-docx

# Small utility
import numpy as np
from flask import Flask, request, jsonify

load_dotenv()  # loads .env into environment

### ---------- CONFIG (from .env) ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")  # e.g. "us-east-1"
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "rag-2")
PINECONE_HOST = os.getenv("PINECONE_HOST")  # optional, e.g. https://rag-2-xxxx.svc...pinecone.io
DRIVE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")  # path to service-account.json
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")  # id of folder to watch
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # text-embedding-004 => 768 dim
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")  # choose model available to your account
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))

if not (GEMINI_API_KEY and PINECONE_API_KEY and DRIVE_SERVICE_ACCOUNT_FILE and DRIVE_FOLDER_ID):
    raise SystemExit("Missing one of required env vars: GEMINI_API_KEY, PINECONE_API_KEY, "
                     "GOOGLE_SERVICE_ACCOUNT_FILE, DRIVE_FOLDER_ID. See README / .env example.")

### ---------- Initialize clients ----------
# Google GenAI client (Gemini / Vertex)
# For Gemini Developer API (API key):
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Pinecone (v6 client)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Use provided host if connecting to an existing serverless index
if PINECONE_HOST:
    index = pc.Index(PINECONE_INDEX, host=PINECONE_HOST)
else:
    # Ensure index exists (serverless)
    existing = pc.list_indexes().names()
    if PINECONE_INDEX not in existing:
        print(f"[Pinecone] Creating index {PINECONE_INDEX} dim={EMBED_DIM} metric=cosine in {PINECONE_ENV or 'us-east-1'}")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV or "us-east-1"),
        )
    index = pc.Index(PINECONE_INDEX)


### ---------- Helper functions (map to n8n nodes) ----------
def download_files_from_drive(target_folder_id: str, output_dir: str = "downloads") -> List[Dict]:
    """
    Maps to: Google Drive File Created + Download File From Google Drive nodes.
    Returns a list of dicts: {"id": fileId, "name": name, "mimeType": mimeType, "path": local_path}
    Notes: share the target folder with the service account email (viewer permission) so it can read.
    """
    os.makedirs(output_dir, exist_ok=True)
    credentials = service_account.Credentials.from_service_account_file(
        DRIVE_SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive = build('drive', 'v3', credentials=credentials, cache_discovery=False)
    q = f"'{target_folder_id}' in parents and trashed=false"
    results = drive.files().list(q=q, fields="files(id,name,mimeType)").execute()
    files = results.get('files', [])
    downloaded = []
    for f in files:
        fid = f['id']
        name = f['name']
        mime = f.get('mimeType', '')
        local_path = os.path.join(output_dir, name)
        # For Google Docs (application/vnd.google-apps.document) we export as plain text
        if mime == 'application/vnd.google-apps.document':
            request = drive.files().export_media(fileId=fid, mimeType='text/plain')
            fh = io.FileIO(local_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.close()
        else:
            request = drive.files().get_media(fileId=fid)
            fh = io.FileIO(local_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.close()
        downloaded.append({"id": fid, "name": name, "mimeType": mime, "path": local_path})
        print(f"[Drive] Downloaded {name} -> {local_path}")
    return downloaded


def extract_text_from_file(path: str, mime: str = "") -> str:
    """Maps to: Default Data Loader + text extraction logic.
       Minimal extractor for .pdf, .docx, .txt, and exported Google Docs (.txt)
    """
    path_lower = path.lower()
    try:
        if path_lower.endswith(".pdf"):
            reader = PdfReader(path)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        elif path_lower.endswith(".docx"):
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            # txt or exported google doc
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
    except Exception as e:
        print(f"[extract_text] Failed to extract {path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Recursive character splitter simplified version (maps to Recursive Character Text Splitter)."""
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, end) if end < text_len else end
    return chunks


def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """Calls Gemini / Vertex embeddings via google-genai SDK.
       Response structure: resp.data[i].embedding
    """
    # batch up to something reasonable if you have very many chunks
    resp = genai_client.embeddings.create(model=model, input=texts)
    embeddings = [d.embedding for d in resp.data]
    return embeddings


def upsert_to_pinecone(chunks: List[str], embeds: List[List[float]], source_meta: Dict):
    """Maps to: Pinecone Vector Store (insert). Upserts list of (id, vector, metadata)."""
    vectors = []
    for c, emb in zip(chunks, embeds):
        vid = str(uuid.uuid4())
        metadata = {"text": c, **source_meta}
        vectors.append({"id": vid, "values": emb, "metadata": metadata})
    # Upsert in batches (Pinecone supports batch upserts)
    print(f"[Pinecone] Upserting {len(vectors)} vectors")
    index.upsert(vectors=vectors)


def ingest_drive_folder_once():
    """
    Full ingestion pipeline for files currently in DRIVE_FOLDER_ID:
    - Download -> extract -> chunk -> embed -> upsert
    """
    files = download_files_from_drive(DRIVE_FOLDER_ID)
    for f in files:
        text = extract_text_from_file(f['path'], f.get('mimeType', ''))
        if not text.strip():
            print(f"[ingest] No text extracted for {f['name']}, skipping.")
            continue
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = embed_texts(chunks)
        meta = {"source_file_id": f['id'], "source_file_name": f['name']}
        upsert_to_pinecone(chunks, embeddings, meta)
    print("[ingest] Done")


### ---------- Retrieval & Chat (Vector Store Tool + AI Agent) ----------
def query_pinecone_and_get_context(query: str, top_k: int = TOP_K):
    q_emb = genai_client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    # modern pinecone query uses vector param 'vector' or 'queries' depending on version
    try:
        res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
        matches = res['matches'] if isinstance(res, dict) and 'matches' in res else getattr(res, 'matches', res)
    except Exception:
        # compatibility attempt
        res = index.query(queries=[q_emb], top_k=top_k, include_metadata=True)
        matches = res['results'][0]['matches']
    # build context string from metadata
    context_pieces = []
    for m in matches:
        md = m.get('metadata', {})
        txt = md.get('text', '')
        # truncate per chunk to avoid massive prompts
        context_pieces.append(f"(score={m.get('score'):.4f}) {txt[:800].strip()}")
    return context_pieces


def ask_gemini_with_context(user_question: str, system_prompt: str, context_chunks: List[str], model: str = CHAT_MODEL):
    """
    Compose a prompt similar to how the n8n AI Agent node used a systemMessage.
    We embed 'relevant documents' into the prompt and send to Gemini chat/generate_content.
    """
    context_text = "\n\n===\n\n".join(context_chunks) if context_chunks else "No relevant context found."
    prompt = (
        system_prompt.strip()
        + "\n\nRelevant documents (short excerpts):\n"
        + context_text
        + "\n\nUser question:\n"
        + user_question
        + "\n\nAnswer concisely, reference any doc excerpts you used."
    )
    response = genai_client.models.generate_content(model=model, contents=prompt)
    # response.text is typical; the SDK returns an object where .text is the generated text
    return getattr(response, "text", str(response))


### ---------- Simple Flask app to test chat -----------
app = Flask(__name__)

# systemMessage from your JSON (AI Agent node)
SYSTEM_MESSAGE = (
    "You are a helpful HR assistant designed to answer employee questions based on company policies. "
    "Retrieve relevant information from the provided internal documents and provide a concise, accurate, and informative answer. "
    "If the answer cannot be found in the provided documents, respond with 'I cannot find the answer in the available resources.'"
)

@app.route("/ingest-now", methods=["POST"])
def http_ingest():
    """Trigger a one-time ingestion from the Drive folder (for demo)."""
    try:
        ingest_drive_folder_once()
        return jsonify({"status": "ok", "message": "Ingestion completed"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/chat", methods=["POST"])
def http_chat():
    payload = request.json or {}
    question = payload.get("query") or payload.get("question")
    if not question:
        return jsonify({"error": "send JSON with {\"query\": \"...\"}"}), 400
    context_chunks = query_pinecone_and_get_context(question, top_k=TOP_K)
    answer = ask_gemini_with_context(question, SYSTEM_MESSAGE, context_chunks)
    return jsonify({"answer": answer, "context_count": len(context_chunks)})

if __name__ == "__main__":
    print("Starting RAG service on http://127.0.0.1:8000")
    app.run(host="0.0.0.0", port=8000)


