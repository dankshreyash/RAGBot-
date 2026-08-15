# Phase 1 - Import
import streamlit as st

# Phase 2 - Import
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Phase 3 - Import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


# Load environment variables
load_dotenv()

# Streamlit title
st.title("NODE RAG CHATBOT")


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


# Create vector store
@st.cache_resource
def get_vectorstore():

    pdf_name = "nodejs.pdf"

    # Load PDF
    loader = PyPDFLoader(pdf_name)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L12-v2"
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# Format retrieved documents
def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


# Chat input
prompt = st.chat_input("Pass your prompt here")


if prompt:

    # Display user message
    st.chat_message("user").markdown(prompt)

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })


    # -----------------------------------
    # Groq LLM
    # -----------------------------------

    model = "llama-3.1-8b-instant"

    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )


    # -----------------------------------
    # RAG Prompt
    # -----------------------------------

    rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer the question based only on the context provided below.

If the answer is not available in the context, say:
"I don't know based on the provided document."

Start the answer directly with no preamble.

Context:
{context}

Question:
{question}
""")


    try:

        # Load vector store
        vectorstore = get_vectorstore()


        if vectorstore is None:

            st.error("Failed to load document")

        else:

            # Create retriever
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 6}
            )


            # -----------------------------------
            # Modern LCEL RAG Chain
            # -----------------------------------

            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | rag_prompt
                | groq_chat
                | StrOutputParser()
            )


            # Get response
            response = rag_chain.invoke(prompt)


            # Display response
            st.chat_message("assistant").markdown(response)


            # Store assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })


    except Exception as e:

        st.error(f"Error: [{str(e)}]")