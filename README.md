# EDUNET INTERNSHIP PROJECT
# BharatDocs – Multilingual Document Q&A System

A simple yet powerful document question‑answering application that lets you upload a PDF and ask questions in **Hindi** or **English**.  
Built with LangChain, Google Gemini, FAISS, and Gradio, it delivers accurate answers by retrieving the most relevant text chunks from your document.

---

## Table of Contents
- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [License](#license)

---
## Demo
![BharatDocs in action](RAG-DEMO.png)
## Features

- **PDF Processing** – Upload a PDF, automatically split into manageable chunks.
- **Multilingual Q&A** – Ask questions in Hindi or English; answers are provided in the same language.
- **Retrieval‑Augmented Generation (RAG)** – Combines vector search with Gemini’s generative capabilities for precise answers.
- **Simple Web Interface** – Powered by Gradio, accessible via browser.
- **Public Share Link** – Launch with a shareable URL for easy demonstration.

---

## Tech Stack

- **LangChain** – Orchestrates the RAG pipeline.
- **Google Generative AI (Gemini)** – Embeddings (`embedding-001`) and chat model (`gemini-1.5-flash`).
- **FAISS** – Local vector store for fast similarity search.
- **PyPDF** – PDF loading and text extraction.
- **Gradio** – Interactive web UI.
- **Python 3.8+** – Core language.

---

## Prerequisites

- Python 3.8 or higher
- A valid **Google API Key** with access to the Generative AI services (Gemini)
- (Optional) A virtual environment for isolated dependency management

---

## Setup & Installation

1. **Clone or download** this repository.

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install langchain langchain-community langchain-google-genai gradio faiss-cpu pypdf python-dotenv
   ```

   > Note: If you have GPU, you may replace `faiss-cpu` with `faiss-gpu`.

4. **Set up your API key**:
   - Create a `.env` file in the project root.
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

5. **Run the application**:
   ```bash
   python app.py
   ```

   The Gradio interface will open in your browser, and a public shareable link will be displayed in the terminal (if `share=True` remains set).

---

## Usage

1. **Upload a PDF** – Click the file input, select a PDF, then press *Process PDF*.  
   The status box will confirm the number of chunks and pages processed.

2. **Ask a question** – Type your query in the text box (in Hindi or English) and press *Ask* or hit Enter.  
   The chatbot will display the answer, retrieved and generated from the document content.

3. **Clear chat** – Use the *Clear Chat* button to reset the conversation.

4. **New document** – Simply upload a new PDF; the previous vector store is replaced.

---

## How It Works

1. **Document Ingestion**  
   - The uploaded PDF is loaded using `PyPDFLoader`.  
   - Text is split into overlapping chunks (1000 characters, 200 overlap) via `RecursiveCharacterTextSplitter`.  
   - Each chunk is converted into a vector embedding using Google’s `embedding-001` model.  
   - The embeddings are stored in a FAISS index.

2. **Question Answering**  
   - The user’s question is embedded using the same model.  
   - FAISS retrieves the top‑3 most similar chunks from the stored index.  
   - The retrieved context, along with the question, is fed into Gemini (`gemini-1.5-flash`) with a custom prompt that instructs the model to respond in the same language as the question.  
   - The answer is returned and displayed in the chat interface.

---

## Project Structure

```
.
├── app.py                 # Main application script
├── .env                   # Environment variables (API key)
├── requirements.txt       # (Optional) List of dependencies
└── README.md              # This file
```

---

## License

This project is created as part of the **IBM SkillsBuild x AICTE Internship** program.  
Feel free to use, modify, and distribute it for educational and non‑commercial purposes.

---

**Built by Avantika Rana** | IBM SkillsBuild x AICTE Internship Project  
*Powered by LangChain, Google Gemini, FAISS, and Gradio*
