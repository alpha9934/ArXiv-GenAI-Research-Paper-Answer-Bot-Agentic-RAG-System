# 🤖 ArXiv-GenAI Research Paper Answer Bot | Agentic RAG System

An advanced, **Agentic Retrieval-Augmented Generation (RAG)** search engine designed to transform static research papers into an interactive, context-aware knowledge base. 

This system simulates an intelligent chatbot for a repository like ArXiv. It is engineered to answer complex questions based on cutting-edge Generative AI research papers while gracefully handling out-of-domain questions via web search.

---

## 🌟 Core Capabilities & Features

This project goes beyond a simple linear RAG pipeline by implementing advanced routing and agentic patterns:

* **Agentic Routing (Corrective RAG Pattern):** Powered by a LangChain ReAct Agent orchestrated via LangGraph. The system intelligently evaluates user queries to decide whether to retrieve context from the local research paper vector database or fallback to a live Wikipedia web search for general knowledge.
* **Transparent Source Attribution:** Solves the LLM "black box" problem. The UI explicitly extracts and displays the exact Document Metadata and "Chunk IDs" used by the LLM to ground its generated responses.
* **Automated Ingestion Pipeline:** Features a self-healing startup sequence. If no vector database is detected, the application automatically loads, chunks, and embeds all PDFs from the `/data` directory into a persistent local FAISS index before launching.
* **Interactive Chat Interface:** A clean, stateful conversational UI built with Streamlit, providing seamless multi-turn interactions and expandable source citations.

---

## 🏗️ Architecture & Tech Stack

This repository demonstrates a modular, enterprise-ready approach to AI application architecture.

* **Orchestration & State Management:** LangGraph
* **Agent Framework:** LangChain (`create_agent`)
* **Vector Database:** FAISS (with persistent local storage)
* **Embeddings Model:** OpenAI (`OpenAIEmbeddings`)
* **Large Language Model:** OpenAI (`ChatOpenAI`)
* **Tools:** Custom Document Retriever Tool, Wikipedia API Wrapper
* **Frontend / UI:** Streamlit
* **Package Management:** `uv`

---

## 🎯 Capstone Milestones Achieved

This project successfully fulfills and exceeds the core requirements and stretch goals of the Analytics Vidhya Capstone:

- [x] **Load and Index Documents:** Built automated ingestion for PDFs using `PyPDFDirectoryLoader`.
- [x] **Experiment with Embeddings & Vector DB:** Implemented persistent FAISS indexing using OpenAI embeddings.
- [x] **Provide Answer Sources:** Engineered metadata extraction to display the top contextual chunks used for generation.
- [x] **Stretch Goal 2 (Application):** Deployed the pipeline into a fully functional Streamlit frontend.
- [x] **Stretch Goal 3 (Agentic RAG):** Enhanced the system with a web-search fallback (Wikipedia) orchestrated by a ReAct agent.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+ installed on your machine.
* An active OpenAI API Key.
* `uv` package manager (recommended for speed, or standard `pip`).

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/RAG-Document-Search-Engine.git
cd RAG-Document-Search-Engine
uv pip install -r requirements.txt
```
*(Ensure `pypdf`, `wikipedia`, and `python-dotenv` are included in your environment).*

### 2. Environment Setup
Create a `.env` file in the root directory and securely add your API key:
```text
OPENAI_API_KEY="sk-proj-your-actual-api-key-here"
```

### 3. Add Research Data
Create a `data/` folder in the root directory of the project. Place your Generative AI research papers (PDFs) into this folder.

### 4. Run the Application
Start the Streamlit server:
```bash
streamlit run streamlit_app.py
```
*On the first run, the system will automatically parse your PDFs, generate vector embeddings, save the FAISS database locally, and launch the chat interface.*