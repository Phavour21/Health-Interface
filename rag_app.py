import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

st.set_page_config(
    page_title="MSME Nigeria RAG Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# API configuration
# ------------------------------------------------------------

CHAT_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

CHAT_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL_NAME = "openai/gpt-oss-20b"

EMBEDDING_BASE_URL = "https://qwen-embed.publicaai.com/v1"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "msme"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .main {
        background-color: #f8fafc;
    }

    /* Header */
    .app-header {
        padding: 1rem 0 0.5rem 0;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        color: #64748b;
        font-size: 1rem;
    }

    /* Chat messages */
    .user-message {
        background-color: #2563eb;
        color: white;
        padding: 12px 16px;
        border-radius: 16px 16px 4px 16px;
        margin: 8px 0;
    }

    .assistant-message {
        background-color: white;
        color: #1e293b;
        padding: 16px;
        border-radius: 16px 16px 16px 4px;
        border: 1px solid #e2e8f0;
        margin: 8px 0;
    }

    /* Sources */
    .source-box {
        background-color: #f1f5f9;
        border-left: 4px solid #2563eb;
        padding: 10px 14px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Buttons */
    .stButton button {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# VALIDATE API KEYS
# ============================================================

if not CHAT_API_KEY:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Add it to your .env file."
    )
    st.stop()

if not EMBEDDING_API_KEY:
    st.error(
        "EMBEDDING_API_KEY is not configured. "
        "Add it to your .env file."
    )
    st.stop()


# ============================================================
# INITIALIZE EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL
    )


# ============================================================
# INITIALIZE CHROMA
# ============================================================

@st.cache_resource
def get_vectorstore():

    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectorstore


# ============================================================
# INITIALIZE RETRIEVER
# ============================================================

@st.cache_resource
def get_retriever():

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10
        }
    )


# ============================================================
# INITIALIZE CHAT MODEL
# ============================================================

@st.cache_resource
def get_chat_model():

    return ChatOpenAI(
        api_key=CHAT_API_KEY,
        base_url=CHAT_BASE_URL,
        model=CHAT_MODEL_NAME,
        temperature=0
    )


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a business consultant providing insights on MSMEs
(Micro, Small and Medium Enterprises) in Nigeria.

You will be provided with retrieved context from the MSME knowledge base.

Context:
{context}

Use ONLY the provided context to answer the user's question.

The context may include information about:

- Understanding MSMEs
- Starting an MSME
- Growing an MSME
- Sustaining an MSME
- Nigerian MSME policies
- Government programmes
- Funding
- Business registration
- Taxation
- Industry-specific information
- Business management

Instructions:

1. Provide a clear and comprehensive answer.
2. Do not invent facts that are not supported by the context.
3. If the answer cannot be found in the context, clearly state that
   the available knowledge base does not contain enough information.
4. Use headings and bullet points where appropriate.
5. Keep the answer professional and practical.
6. If URLs are present in the context, include the relevant URLs.
7. At the end of the response include:

   "To read more, check out this link: [relevant link]"

Question:
{question}
"""
)


# ============================================================
# RAG FUNCTION
# ============================================================

def ask_rag(question):

    retriever = get_retriever()
    chat_model = get_chat_model()

    # Retrieve documents
    documents = retriever.invoke(question)

    # Combine retrieved context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Build prompt
    formatted_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    # Generate response
    response = chat_model.invoke(formatted_prompt)

    # Convert response to string
    answer = StrOutputParser().invoke(response)

    return answer, documents


# ============================================================
# PDF INDEXING
# ============================================================

def index_pdf(uploaded_file):

    # Create temporary PDF
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())

        temp_pdf_path = temp_file.name

    try:

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        # Add metadata
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name

        # Add to Chroma
        vectorstore = get_vectorstore()

        vectorstore.add_documents(chunks)

        return len(chunks)

    finally:

        # Remove temporary file
        Path(temp_pdf_path).unlink(
            missing_ok=True
        )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size: 1.4rem;
            font-weight: 700;
            color: #0f172a;
        ">
            💼 MSME Assistant
        </div>

        <div style="
            color: #64748b;
            margin-top: 5px;
            margin-bottom: 20px;
        ">
            Nigerian MSME Knowledge Assistant
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload an MSME-related PDF to add it to the knowledge base."
    )

    if uploaded_file is not None:

        if st.button(
            "➕ Add PDF to Knowledge Base",
            use_container_width=True
        ):

            with st.spinner("Processing PDF..."):

                try:

                    number_of_chunks = index_pdf(
                        uploaded_file
                    )

                    st.success(
                        f"PDF indexed successfully. "
                        f"{number_of_chunks} chunks added."
                    )

                    # Clear cached retriever so the new
                    # documents become available.
                    get_retriever.clear()

                except Exception as e:

                    st.error(
                        f"Could not index PDF: {str(e)}"
                    )

    st.divider()

    st.subheader("⚙️ Settings")

    st.caption(
        f"Chat model: `{CHAT_MODEL_NAME}`"
    )

    st.caption(
        f"Embedding model: `{EMBEDDING_MODEL_NAME}`"
    )

    st.caption(
        "Retrieval: MMR"
    )

    st.caption(
        "Documents retrieved: 4"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "MSME Nigeria RAG Assistant"
    )

    st.caption(
        "Powered by LangChain + Chroma + Groq"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">

        <div class="app-title">
            💼 MSME Nigeria Assistant
        </div>

        <div class="app-subtitle">
            Ask questions about starting, growing and sustaining
            a business in Nigeria.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.messages) == 0:

    st.info(
        """
        👋 **Welcome!**

        I can help you find information from the MSME knowledge base.

        Try questions such as:

        - What are the steps to start an MSME in Nigeria?
        - What funding options are available to Nigerian MSMEs?
        - How can an MSME improve its business operations?
        - What government policies support MSMEs?
        """
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    content = message["content"]

    with st.chat_message(role):

        st.markdown(
            content,
            unsafe_allow_html=True
        )

        # Display sources for assistant messages
        if (
            role == "assistant"
            and "sources" in message
        ):

            sources = message["sources"]

            if sources:

                with st.expander(
                    f"📚 Sources ({len(sources)})"
                ):

                    for i, source in enumerate(
                        sources,
                        start=1
                    ):

                        metadata = source.metadata

                        source_name = metadata.get(
                            "source",
                            "Unknown source"
                        )

                        page = metadata.get(
                            "page",
                            None
                        )

                        if page is not None:

                            st.markdown(
                                f"""
                                **Source {i}:** {source_name}

                                Page: {page + 1}
                                """
                            )

                        else:

                            st.markdown(
                                f"""
                                **Source {i}:** {source_name}
                                """
                            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about Nigerian MSMEs..."
)


# ============================================================
# HANDLE USER QUESTION
# ============================================================

if question:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user message
    with st.chat_message("user"):

        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the knowledge base..."
        ):

            try:

                answer, documents = ask_rag(
                    question
                )

                # Display answer
                st.markdown(
                    answer,
                    unsafe_allow_html=True
                )

                # Sources
                if documents:

                    with st.expander(
                        f"📚 Sources ({len(documents)})"
                    ):

                        for i, document in enumerate(
                            documents,
                            start=1
                        ):

                            metadata = document.metadata

                            source_name = metadata.get(
                                "source",
                                "Unknown source"
                            )

                            page = metadata.get(
                                "page",
                                None
                            )

                            st.markdown(
                                f"**Source {i}:** "
                                f"{source_name}"
                            )

                            if page is not None:

                                st.caption(
                                    f"Page {page + 1}"
                                )

                            st.markdown(
                                "---"
                            )

                # Save assistant message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": documents
                    }
                )

            except Exception as e:

                error_message = (
                    "Sorry, I encountered an error while "
                    f"processing your question: `{str(e)}`"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )
