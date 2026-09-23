import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (.env)
load_dotenv()

st.set_page_config(page_title="RAG Document Q&A (Gemini)", page_icon="📚", layout="centered")

st.title("📚 Chat with your PDF (Powered by Google Gemini)")
st.caption("Upload a PDF document and ask questions based strictly on its contents.")

# Initialize session states
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def extract_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Use Google Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


# Sidebar for Document Management
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Document"):
        if uploaded_files:
            with st.spinner("Extracting text and generating vector index with Gemini..."):
                raw_text = extract_pdf_text(uploaded_files)
                if raw_text.strip():
                    st.session_state.vector_store = create_vector_store(raw_text)
                    st.success("Document processed and index ready!")
                else:
                    st.error("Could not extract readable text from the PDF.")
        else:
            st.warning("Please upload at least one PDF file first.")


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User query logic
if user_query := st.chat_input("Ask a question about your uploaded document..."):
    if st.session_state.vector_store is None:
        st.error("Please upload and process a document in the sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching context & querying Gemini..."):
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )

                # Initialize Gemini Chat Model
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.5-flash",
                    temperature=0,
                    max_retries=5
                )
                
                system_prompt = (
                    "You are a helpful assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, say that you don't know. "
                    "Use three sentences maximum and keep the answer concise.\n\n"
                    "Context:\n{context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})