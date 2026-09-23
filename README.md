
# 📚 RAG-Powered Knowledge Retrieval System

A high-performance Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, and **Google Gemini**. This system allows users to upload custom PDF documents and perform grounded, contextual Q&A strictly based on the extracted contents.

---

## 📑 Executive Summary

Traditional Large Language Models (LLMs) often suffer from knowledge cutoffs or hallucinations when queried about domain-specific, private, or real-time documents. This application addresses these limitations by pairing **Google Gemini** with vector search techniques (**FAISS**). By converting document text into semantic vector embeddings, the system retrieves only the most relevant passages to answer user queries with high accuracy and strict grounding.

---

## 🏗️ System Architecture & Workflow

```
[ PDF Document ] 
       │
       ▼
[ PyPDF Extractor ] ──► Extracted Plain Text
                               │
                               ▼
               [ Recursive Text Splitter ]
                     (1000 char chunks)
                               │
                               ▼
               [ Google Gemini Embeddings ]
                    (gemini-embedding-001)
                               │
                               ▼
                     [ FAISS Vector Store ]
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
[ User Query ] ──► [ Similarity Retriever ] ──► Relevant Chunks
                                                       │
                                                       ▼
                                            [ Gemini 3.5 Flash LLM ]
                                                       │
                                                       ▼
                                              [ Grounded Answer ]
```

1. **Document Ingestion & Chunking:** PDFs are uploaded via the Streamlit interface and parsed using `pypdf`. The extracted raw text is broken down into overlapping chunks using `RecursiveCharacterTextSplitter` to maintain contextual continuity across chunk boundaries.
2. **Vector Index Generation:** Each text chunk is passed through Google's `gemini-embedding-001` model to generate dense numerical embeddings. These vectors are indexed in-memory using **FAISS** (Facebook AI Similarity Search).
3. **Contextual Retrieval:** When a user submits a query, the system performs cosine similarity search over the vector index to retrieve the top $k$ ($k=3$) most relevant text snippets.
4. **LLM Generation:** The retrieved chunks are formatted into a strict system prompt and sent to `gemini-3.5-flash` alongside the user's question, ensuring answers are concise and tied directly to the context.

---

## 🛠️ Tech Stack & Key Libraries

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit (`>=1.64.0`) | Interactive web interface, file uploader, and chat UI. |
| **LLM Engine** | Google Gemini (`gemini-3.5-flash`) | Core generative model for synthesized reasoning. |
| **Embeddings** | `gemini-embedding-001` | Generates semantic vector representations of text. |
| **Orchestration** | LangChain / LangChain Classic | Handles chains, prompt templates, and retrieval. |
| **Vector Index** | FAISS (`faiss-cpu`) | High-speed local vector similarity searching. |
| **Document Parser**| PyPDF (`pypdf`) | Extracts text streams from PDF files. |
| **Environment** | `python-dotenv` & Streamlit Secrets | Dual handling for local and cloud API credentials. |

---

## ⚙️ Key Implementation Features

### 1. Robust API Authentication
Supports a dual-configuration approach for secure operations across development environments:
* **Local Development:** Loads environment variables seamlessly from a `.gitignore`-shielded `.env` file via `python-dotenv`.
* **Production Deployment:** Dynamically falls back to `st.secrets` when deployed on Streamlit Community Cloud.
* **Validation Defense:** Explicitly passes `google_api_key` to all model class instantiators (`ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings`) to avoid Pydantic runtime initialization errors.

### 2. Built-in Fault Tolerance
* Configured `max_retries=5` on `ChatGoogleGenerativeAI` to handle potential API throughput limits and intermittent `503 UNAVAILABLE` capacity errors seamlessly without crashing the UI.

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
* Python 3.10+
* A valid Google Gemini API Key from Google AI Studio.

### 2. Local Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/PratimaC20/RAG-Powered-Knowledge-Retrieval.git
cd RAG-Powered-Knowledge-Retrieval

# Create and activate virtual environment
python -m venv env4
source env4/bin/activate  # On Windows: .\env4\Scripts\Activate.ps1
```

### 3. Dependencies

Create a clean `requirements.txt` containing:

```text
streamlit
pypdf
langchain
langchain-community
langchain-text-splitters
langchain-google-genai
faiss-cpu
python-dotenv
```

Install requirements:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### 5. Running the Application

Execute the Streamlit application locally:

```bash
streamlit run ragapp.py
```

---

## ☁️ Deployment Strategy (Streamlit Cloud)

1. Push code to GitHub repository (`main` branch).
2. Connect repository to **Streamlit Community Cloud**.
3. Under **App Settings -> Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_actual_gemini_api_key_here"
   ```
4. Deploy app. Streamlit will automatically install dependencies from `requirements.txt` and launch the app.

<img width="1867" height="975" alt="image" src="https://github.com/user-attachments/assets/be596773-9cfd-4fad-b992-0d0c5b5db434" />

<img width="1910" height="1017" alt="image" src="https://github.com/user-attachments/assets/9e83b2fd-a6ec-4eea-ae57-5a2d41a12c6d" />
