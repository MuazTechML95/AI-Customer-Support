# 🤖 SupportAI — Grounded AI Customer Support Assistant

> **A production-oriented RAG customer support assistant built for NovaCart, a fictional e-commerce platform.**

SupportAI is a **Retrieval-Augmented Generation (RAG)** customer support chatbot that answers customer questions using a curated and approved knowledge base. Instead of generating unsupported information, the system retrieves relevant knowledge, provides grounded responses with sources, and safely hands off unsupported questions to human support.

### 🚀 Live Demo

**Try SupportAI live:**
https://ai-customer-support-nhhmfbkqyzt9xh9bpa7epa.streamlit.app/

### ✨ Key Highlights

* 🔎 **Grounded RAG Pipeline** — ingestion → chunking → embeddings → ChromaDB → retrieval → generation
* 🧠 **Semantic Search** using `sentence-transformers/all-MiniLM-L6-v2`
* 📚 **Curated Knowledge Base** covering shipping, returns, payments, warranty, accounts, and cancellations
* 📄 **Source Citations** for knowledge-based answers
* 💬 **Session-Aware Conversations** with follow-up question support
* 🛡️ **Hallucination Protection** with similarity-threshold retrieval
* 🔐 **Prompt-Injection Protection**
* 🔑 **Secret-Leak Detection**
* 🛠️ **Admin Dashboard** for knowledge-base management and index rebuilding
* 📊 **Index Health & Session Metrics**
* 🐳 **Docker Support**
* 🧪 **Automated Tests** for core workflows

### 🎯 Problem

Traditional customer-support chatbots can confidently generate incorrect information or require complex infrastructure.

SupportAI addresses this by combining **RAG, retrieval thresholds, safety guardrails, and controlled knowledge sources** to provide reliable first-line customer support while explicitly deferring unsupported questions to human agents.

The system is designed to handle common queries such as:

* Shipping times and delivery
* Return and refund policies
* Warranty coverage
* Payment methods
* Account policies
* Order cancellation
* Product information

### 🏗️ System Architecture

```text
User
  │
  ▼
Streamlit UI
  │
  ▼
Support Service
  │
  ▼
Safety & Guardrails
  │
  ▼
Session Manager
  │
  ▼
Retriever
  │
  ▼
Chroma Vector Store
  │
  ▼
Embeddings
  │
  ▼
Grounded LLM Generation
  │
  ├──► Answer + Sources
  │
  └──► Safe Unsupported-Query Response
```

### 🔄 RAG Pipeline

```text
Knowledge Base
      │
      ▼
Document Ingestion
      │
      ▼
Text Cleaning & Chunking
      │
      ▼
Sentence Transformers
(all-MiniLM-L6-v2)
      │
      ▼
Chroma Vector Store
      │
      ▼
Similarity Retrieval
      │
      ▼
Relevant Context
      │
      ▼
LLM
      │
      ▼
Grounded Response + Sources
```

The project uses approximately **800-character chunks with 100-character overlap**, with configurable chunking parameters.

### 🛡️ Responsible AI & Safety

SupportAI is designed to avoid unsupported answers rather than simply relying on an LLM prompt.

If retrieved content does not pass the configured similarity threshold, the system returns a safe fallback response **without calling the LLM**. It also includes pre-LLM prompt-injection detection and post-LLM secret-leak scanning.

### 📚 Knowledge Base

Knowledge is maintained as Markdown documents with YAML front matter:

```markdown
---
title: Return & Refund Policy
category: policy
doc_id: return_refund_policy
---

# Return & Refund Policy

...
```

The knowledge base is organized into:

```text
data/
└── knowledge_base/
    ├── faqs/
    ├── policies/
    └── products/
```

New business knowledge can be added using the same format without changing the core application code.

### 🛠️ Tech Stack

| Component        | Technology                     |
| ---------------- | ------------------------------ |
| UI               | Streamlit                      |
| Language         | Python                         |
| RAG              | Retrieval-Augmented Generation |
| Embeddings       | Sentence Transformers          |
| Embedding Model  | `all-MiniLM-L6-v2`             |
| Vector Database  | ChromaDB                       |
| LLM              | OpenAI / Anthropic compatible  |
| Configuration    | `.env` / Streamlit Secrets     |
| Deployment       | Streamlit Cloud                |
| Containerization | Docker                         |
| Testing          | Pytest                         |

### 🧪 Testing

The project includes tests covering:

* Grounded retrieval
* Unsupported questions
* Follow-up conversation context
* Prompt-injection resistance
* Session isolation

The project intentionally does **not claim formal RAG accuracy or latency benchmarks**, because a formal evaluation framework such as RAGAS has not yet been run.

### ⚙️ Local Setup

```bash
git clone <repository>
cd ai-customer-support

python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables:

```bash
cp .env.example .env
```

Set:

```text
LLM_PROVIDER
LLM_MODEL
LLM_API_KEY
```

Build the knowledge index:

```bash
python -m app.rag.vector_store
```

Run the application:

```bash
streamlit run app/ui/streamlit_app.py
```

Or:

```bash
python run.py
```

### 📁 Project Structure

```text
ai-customer-support/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── rag/
│   ├── chat/
│   ├── safety/
│   ├── services/
│   └── ui/
│
├── data/
│   └── knowledge_base/
│
├── vectorstore/
├── tests/
├── screenshots/
│
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── README.md
└── run.py
```

### 🔮 Current Limitations

* Session history is currently stored in memory and is cleared after restart.
* Similarity threshold is heuristic and requires tuning for real-world usage.
* No authentication or customer account integration.
* Current knowledge base and prompts are English-only.
* The system has not been load-tested.
* No formal RAGAS evaluation has been performed yet.

These limitations are intentionally documented rather than presenting the prototype as a fully production-scale system.

### 🌐 Live Demo

👉 **[Launch SupportAI](https://ai-customer-support-nhhmfbkqyzt9xh9bpa7epa.streamlit.app/)**

**SupportAI — grounded answers, controlled knowledge, and safe customer support.**
