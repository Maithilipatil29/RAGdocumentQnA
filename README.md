<div align="center">

# 📚 RAG Document Q&A

### Chat with your PDFs. Find answers. Understand research faster.

A simple Generative AI application that lets you upload PDF documents, ask questions about their content, and generate document summaries using **Retrieval-Augmented Generation (RAG)**.

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036?style=for-the-badge)

</div>

---

## ✨ Overview

Reading lengthy research papers and technical documents can take time. **RAG Document Q&A** makes documents easier to explore by combining document retrieval with a Large Language Model (LLM).

Upload one or more PDFs, process their content, and ask questions in natural language. The application retrieves relevant text from the uploaded documents and supplies it to the language model to generate a context-based response. You can also generate a summary of the document content.

> **Project goal:** Explore the practical foundations of Generative AI, LLM integration, semantic search, and Retrieval-Augmented Generation.

## 🚀 Features

- 📄 **Multiple PDF uploads** — provide one or more PDF documents.
- 💬 **Document question answering** — ask questions in natural language.
- 🔎 **Context retrieval** — search for relevant document chunks to help answer questions.
- 🧠 **Semantic embeddings** — represent text as vectors for similarity search.
- ⚡ **FAISS vector search** — retrieve relevant chunks from the processed documents.
- 📝 **Document summarization** — generate a summary from document chunks in batches.
- 👀 **Retrieved context view** — inspect the text retrieved for a question.
- 🖥️ **Interactive web interface** — use the app through Streamlit.

## 🧭 How It Works

### Question-answering workflow

```text
       ┌────────────────────┐
       │   Upload PDF(s)    │
       └─────────┬──────────┘
                 ▼
       ┌────────────────────┐
       │ Extract PDF Text   │
       └─────────┬──────────┘
                 ▼
       ┌────────────────────┐
       │ Split into Chunks  │
       └─────────┬──────────┘
                 ▼
       ┌────────────────────┐
       │ Generate Embeddings│
       └─────────┬──────────┘
                 ▼
       ┌────────────────────┐
       │ FAISS Vector Store │
       └─────────┬──────────┘
                 │
       User Question
                 ▼
       ┌────────────────────┐
       │ Retrieve Relevant  │
       │      Chunks        │
       └─────────┬──────────┘
                 ▼
       ┌────────────────────┐
       │ Context + Question │
       │    Sent to LLM     │
       └─────────┬──────────┘
                 ▼
       ┌────────────────────┐
       │ Answer in Streamlit│
       └────────────────────┘
```

### What is RAG?

**Retrieval-Augmented Generation (RAG)** combines two steps:

1. **Retrieval:** Find relevant passages in your uploaded documents.
2. **Generation:** Give those passages and the user's question to an LLM so it can generate an answer grounded in the retrieved context.

This approach helps the application answer questions about user-provided documents without requiring the entire document to be included in every question.

## 🛠️ Tech Stack

| Technology | Role in the project |
|---|---|
| **Python** | Core application language |
| **Streamlit** | Interactive web UI |
| **LangChain** | Document and LLM integration |
| **PyPDF / PyPDFLoader** | Extract text from PDFs |
| **RecursiveCharacterTextSplitter** | Split document text into manageable chunks |
| **Hugging Face Sentence Transformers** | Create text embeddings |
| **FAISS** | Store and search vectors |
| **Groq** | Hosted LLM inference |
| **python-dotenv** | Load configuration from environment variables |

### Models

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** Configured through `GROQ_MODEL` in your environment file. For example, `openai/gpt-oss-120b` if it is available to your Groq account.

## 📂 Repository Structure

```text
RAGdocumentQnA/
├── app.py          # Streamlit app and RAG workflow
├── .gitignore      # Files and secrets excluded from Git
└── README.md       # Project documentation
```

> This diagram reflects the files currently in the repository. If you later split the application into modules, update this section to match the new structure.

## ⚙️ Getting Started

### 1. Prerequisites

Make sure you have:

- Python installed
- A Groq API key
- Git (optional, for cloning the repository)

### 2. Clone the repository

Replace `<your-username>` with your GitHub username and use your actual repository name.

```bash
git clone https://github.com/<your-username>/RAGdocumentQnA.git
cd RAGdocumentQnA
```

### 3. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you already use a virtual environment outside this repository, activate that environment instead.

### 4. Install dependencies

Install the packages imported by your `app.py`. If your repository has a `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

If you have not added `requirements.txt` yet, create one with the dependencies used by your app before sharing the project so other developers can reproduce your environment.

### 5. Configure your API key

Create a `.env` file in the location expected by your app and add:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Use the model name supported by your Groq account. If your code uses `LLM_PROVIDER`, you can also configure it as expected by the application.

🔐 **Security:** Never commit your real API key. Keep `.env` in `.gitignore`. If a key is accidentally published, revoke it and create a replacement.

### 6. Run the app

From the directory containing `app.py`, run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL to open in your browser.

## 🧑‍💻 How to Use

1. Open the application in your browser.
2. Upload one or more PDF documents.
3. Click **Process Documents** and wait for extraction, chunking, embeddings, and indexing to finish.
4. Select **Ask a Question** and enter a question about the uploaded content.
5. Review the generated answer and inspect the retrieved context if available.
6. Select **Summarize Documents** to generate a document summary.

### Example questions

- What is the main objective of this paper?
- Which methodology does the author use?
- What dataset is used in the study?
- Which models or techniques are discussed?
- What are the key findings and conclusions?
- What limitations are mentioned?

## 🧠 Key Concepts Practiced

- Generative AI and Large Language Models
- Retrieval-Augmented Generation (RAG)
- PDF ingestion and text extraction
- Text chunking and overlap
- Embeddings and semantic similarity
- Vector search with FAISS
- Prompt engineering
- LLM API integration
- Batch-based document summarization
- Building an interactive application with Streamlit
- Managing API keys through environment variables

## ⚠️ Notes & Limitations

- The application is designed for text-based PDFs. Scanned PDFs may require OCR.
- Answer quality depends on PDF text extraction, chunking, retrieval quality, and the selected LLM.
- LLM output can still be incorrect; verify important information against the original document.
- Large documents may require multiple summary batches and can be limited by model rate limits or token limits.
- The vector store may be recreated when documents are processed; persistent storage depends on the current implementation.
- Do not upload confidential or sensitive documents unless you understand the privacy and data-handling policies of the services you use.

## 🗺️ Future Improvements

- [ ] Show source PDF names and page numbers beside answers
- [ ] Add support for DOCX and TXT files
- [ ] Preserve the vector index between sessions
- [ ] Add chat history for follow-up questions
- [ ] Add configurable chunk size and retrieval count
- [ ] Improve error handling and loading feedback
- [ ] Add RAG evaluation for answer relevance and faithfulness
- [ ] Deploy the application online

## 🖼️ Screenshots

Add screenshots of your running application here to make the repository easier to explore.

For example:

```markdown
![Application home screen](screenshots/home.png)
![Question answering example](screenshots/question-answer.png)
```

Create a `screenshots/` folder and add your own screenshots before using these image links.

## 👩‍💻 Author

**Maithili Patil**

B.Tech — Computer Science (Data Science)

Interested in AI/ML, Generative AI, LLM applications, and Retrieval-Augmented Generation.

---

<div align="center">

**Built as a hands-on project to learn and apply Generative AI concepts.**

⭐ If you find this project useful, consider giving the repository a star!

</div>
