import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Global vectorstore
vectorstore = None

def process_pdf(pdf_file):
    """Load PDF, split into chunks, store in ChromaDB"""
    global vectorstore

    if pdf_file is None:
        return "Please upload a PDF file."

    try:
        tmp_path = pdf_file

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        # Create embeddings and store in FAISS
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        return f"✅ PDF processed successfully! {len(chunks)} chunks created from {len(documents)} pages. You can now ask questions."

    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"


def answer_question(question, history):
    """Answer question using RAG pipeline"""
    global vectorstore

    if vectorstore is None:
        return history + [[question, "⚠️ Please upload and process a PDF first."]]

    if not question.strip():
        return history

    try:
        # Setup LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        # Custom prompt for bilingual support
        prompt_template = """You are BharatDocs, a helpful multilingual document assistant.
Use the following context from the document to answer the question.
Answer in the same language as the question (Hindi or English).
If you don't know the answer from the context, say so clearly.
Keep your answer concise and helpful.

Context: {context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Get answer
        result = qa_chain.invoke({"query": question})
        answer = result["result"]

        return history + [[question, answer]]

    except Exception as e:
        return history + [[question, f"❌ Error: {str(e)}"]]


# Gradio UI
with gr.Blocks(
    title="BharatDocs",
    theme=gr.themes.Soft(),
    css="""
        .container { max-width: 800px; margin: auto; }
        .title { text-align: center; color: #4A90E2; }
    """
) as demo:

    gr.Markdown("""
    # 📄 BharatDocs
    ### Multilingual Document Q&A System
    Upload any PDF and ask questions in **Hindi or English** — powered by Google Gemini & RAG
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="📂 Upload PDF Document",
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("🚀 Process PDF", variant="primary")
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Upload a PDF to get started..."
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 Chat with your Document",
                height=400
            )
            with gr.Row():
                question_input = gr.Textbox(
                    label="Ask a question (Hindi or English)",
                    placeholder="e.g. What is this document about? / यह दस्तावेज़ किस बारे में है?",
                    scale=4
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

    gr.Markdown("""
    ---
    **Built by Avantika Rana** | IBM SkillsBuild x AICTE Internship Project
    *Powered by LangChain • ChromaDB • Google Gemini • Gradio*
    """)

    # Event handlers
    upload_btn.click(
        fn=lambda f: process_pdf(f.name if f else None),
        inputs=[pdf_input],
        outputs=[upload_status]
    )

    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        outputs=[question_input]
    )

    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        outputs=[question_input]
    )

    clear_btn.click(
        fn=lambda: [],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch(share=True)
