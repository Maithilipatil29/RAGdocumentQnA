import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

groq_api_key = os.getenv("GROQ_API_KEY")

groq_model = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


if not groq_api_key:

    st.error(
        f"GROQ_API_KEY not found.\n\n"
        f"Expected .env file at:\n{ENV_PATH}"
    )

    st.stop()


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# 3. INITIALIZE LLM
# ============================================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model=groq_model,
    temperature=0
)


# ============================================================
# 4. PROMPTS
# ============================================================

question_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer cannot be found in the context, say:

"I could not find the answer in the provided documents."

Do not make up information.

<context>
{context}
</context>

Question:
{question}

Answer:
"""
)


summary_chunk_prompt = ChatPromptTemplate.from_template(
    """
You are summarizing part of a research paper.

Summarize the following section accurately.

Focus on:
- Main objective
- Problem being addressed
- Methodology
- Dataset/data
- Models or techniques
- Important findings
- Conclusion or observations

Use ONLY the provided content.
Do not invent information.

Research paper section:
{context}

Summary:
"""
)

final_summary_prompt = ChatPromptTemplate.from_template(
    """
You are a research paper summarization assistant.

Below are summaries of different sections of a research paper.

Combine them into one clear and coherent final summary.

Include, when available:

1. Main objective
2. Problem being addressed
3. Methodology
4. Dataset/data
5. Models and techniques
6. Important results/findings
7. Conclusion

Do not add information that is not present in the summaries.

Section summaries:
{context}

Final Summary:
"""
)


# ============================================================
# 5. CREATE EMBEDDINGS
# ============================================================

def create_vector_database(uploaded_files):

    all_documents = []

    # --------------------------------------------------------
    # Save uploaded files temporarily and load them
    # --------------------------------------------------------

    temp_directory = Path("temp_documents")

    temp_directory.mkdir(
        exist_ok=True
    )

    for uploaded_file in uploaded_files:

        file_path = (
            temp_directory /
            uploaded_file.name
        )

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        # Load PDF
        loader = PyPDFLoader(
            str(file_path)
        )

        documents = loader.load()

        all_documents.extend(
            documents
        )

    # --------------------------------------------------------
    # Check documents
    # --------------------------------------------------------

    if not all_documents:

        st.error(
            "No documents could be loaded."
        )

        return None

    # --------------------------------------------------------
    # Text splitting
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    final_documents = (
        text_splitter.split_documents(
            all_documents
        )
    )

    # --------------------------------------------------------
    # Create embedding model
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # Create FAISS vector database
    # --------------------------------------------------------

    vectors = FAISS.from_documents(
        final_documents,
        embeddings
    )

    return vectors, final_documents

def generate_document_summary(documents):

    # --------------------------------------------------------
    # Step 1: Divide documents into small batches
    # --------------------------------------------------------

    batch_size = 6

    summaries = []

    for i in range(
        0,
        len(documents),
        batch_size
    ):

        batch = documents[
            i:i + batch_size
        ]

        batch_context = "\n\n".join(
            document.page_content
            for document in batch
        )

        # ----------------------------------------------------
        # Create prompt for this batch
        # ----------------------------------------------------

        formatted_prompt = (
            summary_chunk_prompt.invoke(
                {
                    "context": batch_context
                }
            )
        )

        # ----------------------------------------------------
        # Ask LLM to summarize this batch
        # ----------------------------------------------------

        response = llm.invoke(
            formatted_prompt
        )

        summaries.append(
            response.content
        )

    # --------------------------------------------------------
    # Step 2: Combine the smaller summaries
    # --------------------------------------------------------

    combined_summaries = "\n\n".join(
        f"Section {i + 1}:\n{summary}"
        for i, summary in enumerate(summaries)
    )

    # --------------------------------------------------------
    # Step 3: If combined summaries are still large,
    # summarize them in batches again
    # --------------------------------------------------------

    summary_documents = [
        Document(
            page_content=summary
        )
        for summary in summaries
    ]

    if len(combined_summaries) > 20000:

        final_parts = []

        for i in range(
            0,
            len(summary_documents),
            5
        ):

            batch = summary_documents[
                i:i + 5
            ]

            batch_context = "\n\n".join(
                document.page_content
                for document in batch
            )

            formatted_prompt = (
                summary_chunk_prompt.invoke(
                    {
                        "context": batch_context
                    }
                )
            )

            response = llm.invoke(
                formatted_prompt
            )

            final_parts.append(
                response.content
            )

        combined_summaries = "\n\n".join(
            final_parts
        )

    # --------------------------------------------------------
    # Step 4: Generate final summary
    # --------------------------------------------------------

    final_prompt = final_summary_prompt.invoke(
        {
            "context": combined_summaries
        }
    )

    final_response = llm.invoke(
        final_prompt
    )

    return final_response.content

# ============================================================
# 6. HEADER
# ============================================================

st.title("📚 RAG Document Q&A")

st.write(
    """
Upload one or more PDF documents and ask questions
about their content using Retrieval-Augmented Generation.
"""
)


# ============================================================
# 7. FILE UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# 8. DISPLAY UPLOADED FILES
# ============================================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} document(s) uploaded."
    )

    for file in uploaded_files:

        st.write(
            f"📄 {file.name}"
        )


# ============================================================
# 9. PROCESS DOCUMENTS
# ============================================================

if st.button(
    "🔍 Process Documents"
):

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF document."
        )

    else:

        with st.spinner(
            "Loading documents, creating chunks and embeddings..."
        ):

            result = create_vector_database(
                uploaded_files
            )

        if result:

            vectors, final_documents = result

            st.session_state.vectors = vectors

            st.session_state.documents = (
                final_documents
            )

            st.session_state.uploaded_files = [
                file.name
                for file in uploaded_files
            ]

            st.success(
                "Documents processed successfully!"
            )

            st.info(
                f"Created {len(final_documents)} text chunks."
            )


# ============================================================
# 10. CHECK WHETHER DOCUMENTS ARE PROCESSED
# ============================================================

if "vectors" in st.session_state:

    st.divider()

    st.subheader(
        "Ask questions about your documents"
    )

    # --------------------------------------------------------
    # Operation selection
    # --------------------------------------------------------

    operation = st.radio(
        "Choose an operation:",
        [
            "Ask a Question",
            "Summarize Documents"
        ],
        horizontal=True
    )


    # ========================================================
    # 11. QUESTION ANSWERING
    # ========================================================

    if operation == "Ask a Question":

        user_question = st.text_input(
            "Enter your question:"
        )

        if st.button(
            "💬 Get Answer"
        ):

            if not user_question:

                st.warning(
                    "Please enter a question."
                )

            else:

                # ------------------------------------------------
                # Retrieve relevant chunks
                # ------------------------------------------------

                retriever = (
                    st.session_state.vectors
                    .as_retriever(
                        search_kwargs={
                            "k": 5
                        }
                    )
                )

                start = time.process_time()

                retrieved_documents = (
                    retriever.invoke(
                        user_question
                    )
                )

                # ------------------------------------------------
                # Combine retrieved chunks
                # ------------------------------------------------

                context = "\n\n".join(
                    document.page_content
                    for document
                    in retrieved_documents
                )

                # ------------------------------------------------
                # Create prompt
                # ------------------------------------------------

                formatted_prompt = (
                    question_prompt.invoke(
                        {
                            "context": context,
                            "question": user_question
                        }
                    )
                )

                # ------------------------------------------------
                # Generate answer
                # ------------------------------------------------

                response = llm.invoke(
                    formatted_prompt
                )

                response_time = (
                    time.process_time()
                    - start
                )

                # ------------------------------------------------
                # Display answer
                # ------------------------------------------------

                st.subheader(
                    "Answer"
                )

                st.write(
                    response.content
                )

                st.caption(
                    f"Response time: "
                    f"{response_time:.2f} seconds"
                )

                # ------------------------------------------------
                # Retrieved context
                # ------------------------------------------------

                with st.expander(
                    "🔎 View Retrieved Context"
                ):

                    for i, document in enumerate(
                        retrieved_documents,
                        start=1
                    ):

                        st.markdown(
                            f"### Chunk {i}"
                        )

                        st.write(
                            document.page_content
                        )

                        st.divider()


    # ========================================================
    # 12. DOCUMENT SUMMARY
    # ========================================================

    else:

        st.write(
            "Generate a summary of the processed documents."
        )

        if st.button(
            "📝 Generate Summary"
        ):

            documents = (
                st.session_state.documents
            )

            with st.spinner(
                "Generating document summary..."
            ):

                start = time.process_time()

                summary = generate_document_summary(
                    documents
                )

                response_time = (
                    time.process_time()
                    - start
                )

            st.subheader(
                "📄 Document Summary"
            )

            st.write(
                summary
            )

            st.caption(
                f"Response time: "
                f"{response_time:.2f} seconds"
            )
            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if len(context) > 50000:

                st.warning(
                    "The document is large. "
                    "The summary will use the first "
                    "50,000 characters."
                )

                context = context[:50000]

            # ------------------------------------------------
            # Create summary prompt
            # ------------------------------------------------

            formatted_prompt = (
                summary_prompt.invoke(
                    {
                        "context": context
                    }
                )
            )

            # ------------------------------------------------
            # Generate summary
            # ------------------------------------------------

            start = time.process_time()

            response = llm.invoke(
                formatted_prompt
            )

            response_time = (
                time.process_time()
                - start
            )

            # ------------------------------------------------
            # Display summary
            # ------------------------------------------------

            st.subheader(
                "📄 Document Summary"
            )

            st.write(
                response.content
            )

            st.caption(
                f"Response time: "
                f"{response_time:.2f} seconds"
            )