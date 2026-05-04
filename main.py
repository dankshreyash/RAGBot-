# phase 1 import
import streamlit as st

# phase 2 import
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# phase 3 import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

load_dotenv()
st.title("RAG CHATBOT")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_name = "nodejs.pdf"

    # Load PDF
    loader = PyPDFLoader(pdf_name)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Embed and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Input
prompt = st.chat_input("pass your prompt here")

if prompt:
    # Show user message
    st.chat_message("user").markdown(prompt)

    # Store message
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })

    model = "llama-3.1-8b-instant"
    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )

    rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context provided below.
Start the answer directly with no preamble.

Context:
{context}

Question: {question}
""")

    try:
        vectorstore = get_vectorstore()

        if vectorstore is None:
            st.error("Failed to load document")
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # Modern LCEL RAG chain (replaces deprecated RetrievalQA)
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | rag_prompt
                | groq_chat
                | StrOutputParser()
            )

            response = rag_chain.invoke(prompt)

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error: [{str(e)}]")