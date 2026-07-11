import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_groq import ChatGroq
import tempfile
import os

st.set_page_config(page_title="PDF Query RAG", page_icon="📄")
st.title("📄 PDF Query RAG System")
st.write("Upload a PDF and ask questions about it!")

# Load secrets (from Streamlit secrets, not hardcoded)
ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Cache embedding model (loads once)
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("Processing PDF..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and split
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # Store in AstraDB
        vector_store = AstraDBVectorStore(
            embedding=embedding_model,
            collection_name="pdf_rag_streamlit",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
        )
        vector_store.add_documents(chunks)
        st.session_state.vector_store = vector_store
        os.remove(tmp_path)

    st.success(f"PDF processed! {len(chunks)} chunks stored in AstraDB.")

# Chat interface
if st.session_state.vector_store is not None:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask a question about your PDF")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])

                prompt = f"""Answer the question based only on the following context.
If the answer is not in the context, say "I don't have enough information in the document to answer this."

Context:
{context}

Question: {query}

Answer:"""
                response = llm.invoke(prompt)
                st.write(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
else:
    st.info("Please upload a PDF to get started.")
