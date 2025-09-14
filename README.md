
# 📄 RAG Pipeline with Google Drive + Gemini + Pinecone + Flask

A simple **Retrieval-Augmented Generation (RAG)** pipeline that ingests documents from **Google Drive**, stores them in **Pinecone** using **Gemini embeddings**, and exposes a **Flask API** for retrieval-based chat.

---

## 🚀 Features

* **Google Drive Ingestion**
  Poll a shared Google Drive folder (via service account) and download files.

* **Text Extraction**
  Extracts text from:

  * PDF (`.pdf`)
  * Word (`.docx`)
  * Text (`.txt`)
  * Google Docs (exported as plain text)

* **Chunking**
  Splits large documents into overlapping chunks for better retrieval.

* **Embeddings with Gemini**
  Uses **Google GenAI (Gemini)** embedding model (`text-embedding-004`).

* **Vector Storage with Pinecone**
  Stores embeddings with metadata and supports semantic search.

* **Chat Endpoint**
  Retrieves relevant context and asks **Gemini chat model** (`gemini-2.0-flash`) to generate an answer.

---

## 🛠️ Requirements

* Python 3.9+
* Google Cloud Service Account (with Drive API access)
* Gemini API Key (Google AI Studio / Vertex AI)
* Pinecone account & API key

---

## 📦 Installation

1. **Clone this repo**

```bash
git clone https://github.com/your-username/rag-pipeline.git
cd rag-pipeline
```

2. **Create virtual environment & install dependencies**

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up environment variables**
   Create a `.env` file in the project root:

```ini
# Gemini API
GEMINI_API_KEY=your_gemini_api_key

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=rag-2
# Optional: use existing host if available
# PINECONE_HOST=https://rag-2-xxxx.svc.us-east-1-aws.pinecone.io

# Google Drive
GOOGLE_SERVICE_ACCOUNT_FILE=service-account.json
DRIVE_FOLDER_ID=your_drive_folder_id

# Models & Chunking
EMBED_MODEL=text-embedding-004
EMBED_DIM=768
CHAT_MODEL=gemini-2.0-flash
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
```

4. **Enable Google Drive API**

   * Create a service account in Google Cloud
   * Download `service-account.json` and place it in the project root
   * Share the target Drive folder with the service account email (viewer access)

---

## ▶️ Running the Service

Start the Flask app:

```bash
python app.py
```

By default, it runs on:

```
http://127.0.0.1:8000
```

---

## 📡 API Endpoints

### 🔄 Ingest Drive folder

**POST** `/ingest-now`
Triggers ingestion of all files from the configured Drive folder.

```bash
curl -X POST http://127.0.0.1:8000/ingest-now
```

Response:

```json
{
  "status": "ok",
  "message": "Ingestion completed"
}
```

---

### 💬 Chat with Documents

**POST** `/chat`
Send a query to retrieve relevant context and generate an answer.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the company leave policy?"}'
```

Response:

```json
{
  "answer": "Employees are entitled to 20 paid leaves per year...",
  "context_count": 3
}
```

---

## 📂 Project Structure

```
.
├── app.py                  # Main Flask application
├── requirements.txt        # Dependencies
├── README.md               # Documentation
├── .env                    # Environment variables
├── downloads/              # Downloaded files from Drive
└── service-account.json    # Google service account key
```

---

## 🔧 Customization

* Change **system prompt** in `app.py` (`SYSTEM_MESSAGE`) to adapt for other domains (e.g., legal, medical, education).
* Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env` for optimal retrieval.
* Replace `gemini-2.0-flash` with another available Gemini chat model.

---

## ⚠️ Notes

* Make sure the service account has **Drive API access** and permission to the folder.
* For large document sets, consider batching ingestion.
* This is a **beginner-friendly RAG pipeline**, not production-hardened.

---

## 📜 License

MIT License – feel free to use and modify.

---
