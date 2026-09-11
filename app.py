import os
import requests
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Modern LCEL Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PDF_URL = "https://www.na.gov.pk/uploads/documents/1470910659_707.pdf"
LOCAL_PDF = "PECA_2016.pdf"

st.set_page_config(page_title="Cyberlawbook - Pakistan PECA RAG", page_icon="⚖️", layout="wide")
st.title("⚖️ Cyberlawbook")
st.caption("RAG Legal Assistant for Pakistan's Prevention of Electronic Crimes Act (PECA 2016)")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ App Settings")

secret_api_key = st.secrets.get("GROQ_API_KEY", "")
groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password", value=secret_api_key)

technicality = st.sidebar.select_slider(
    "Technicality / Explanation Style",
    options=["Simple (Layman terms)", "Balanced (Standard)", "Legal Expert (Formal analysis)"],
    value="Balanced (Standard)"
)

response_size = st.sidebar.radio(
    "Response Length",
    options=["Concise (Brief summary)", "Detailed (Comprehensive elaboration)"],
    index=0
)

include_penalties = st.sidebar.checkbox("Highlight Penalties & Punishment Details", value=True)

# --- Embedding & Retrieval Setup ---
@st.cache_resource(show_spinner="Downloading PECA PDF & building vector index...")
def init_vector_store():
    if not os.path.exists(LOCAL_PDF):
        res = requests.get(PDF_URL, timeout=30)
        with open(LOCAL_PDF, "wb") as f:
            f.write(res.content)
            
    loader = PyPDFLoader(LOCAL_PDF)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

if not groq_api_key:
    st.info("👈 Please enter your Groq API Key in the sidebar or Secrets to proceed.")
    st.stop()

try:
    vectorstore = init_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
except Exception as e:
    st.error(f"Failed to initialize vector database: {e}")
    st.stop()

# --- System Prompt Formulation ---
system_prompt_str = f"""
You are an AI legal expert specializing in Pakistan's Cyber Laws, specifically the Prevention of Electronic Crimes Act (PECA 2016).
Answer the user's question accurately using ONLY the retrieved legal context.
If the answer is not contained within the provided context, state clearly that the document does not contain this specific provision.

Strict Guidelines:
- Explanation Level: {technicality}
- Detail Level: {response_size}
- Highlight Penalties/Imprisonment: {"Yes, explicitly mention relevant imprisonment terms and fines if mentioned." if include_penalties else "No"}
- Always reference section numbers or titles when present in the context.

Context:
{{context}}

Question:
{{input}}
"""

prompt = ChatPromptTemplate.from_template(system_prompt_str)

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question about Pakistani Cyber Law (PECA 2016)..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing PECA 2016 provisions..."):
            try:
                llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0.2)
                
                # Fetch retrieved documents
                retrieved_docs = retriever.invoke(user_query)
                context_str = format_docs(retrieved_docs)
                
                # Build LCEL Chain
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context_str, "input": user_query})
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                with st.expander("🔍 View Retrieved Legal Context"):
                    for idx, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Source Chunk {idx+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                        st.text(doc.page_content)
            except Exception as e:
                st.error(f"Error executing request: {e}")
