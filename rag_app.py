"""
Multi-PDF RAG app
- HuggingFace embeddings (local, free) + FAISS vector store
- OpenAI gpt-4o-mini for answers, wrapped with LangSmith for tracing
Run:  streamlit run rag_app.py
Env:  put OPENAI_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING in a .env file
"""
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

# ---------- Setup ----------
load_dotenv()  # reads OPENAI_API_KEY, LANGSMITH_* from .env
st.set_page_config(page_title="Multi-PDF RAG", page_icon="📄")
st.title("📄 Ask your PDFs")

if not os.getenv("OPENAI_API_KEY"):
    st.error("Set the OPENAI_API_KEY environment variable first.")
    st.stop()

# LangSmith wrapper: every OpenAI call gets traced automatically
client = wrap_openai(OpenAI())


@st.cache_resource
def get_embeddings():
    # Loaded once, reused across reruns
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------- Step 1: Upload + index ----------
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if files:
    names = sorted(f.name for f in files)

    # Rebuild only when the set of uploaded files changes
    if st.session_state.get("indexed_files") != names:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = []

        with st.spinner("Reading and indexing PDFs..."):
            for f in files:
                # Save upload to a temp file so PyPDFLoader can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    path = tmp.name

                pages = PyPDFLoader(path).load()  # one Document per page
                for p in pages:
                    p.metadata["source"] = f.name               # real file name, not temp path
                    p.metadata["page"] = p.metadata["page"] + 1  # 0-based -> human page number

                chunks.extend(splitter.split_documents(pages))  # metadata carries over
                os.remove(path)

            st.session_state.store = FAISS.from_documents(chunks, get_embeddings())
            st.session_state.indexed_files = names
            st.session_state.chunk_count = len(chunks)

    st.write("**Uploaded:** " + ", ".join(names))
    st.success(f"Indexed {st.session_state.chunk_count} chunks.")


# ---------- Step 2: Retrieve + answer ----------
@traceable(name="retrieve")
def retrieve(question, k=4):
    return st.session_state.store.as_retriever(search_kwargs={"k": k}).invoke(question)


@traceable(name="answer")
def answer(question, docs):
    context = "\n\n".join(
        f"[{d.metadata['source']} - page {d.metadata['page']}]\n{d.page_content}" for d in docs
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Answer only from the context. "
                                          "If the answer isn't there, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


if "store" in st.session_state:
    question = st.text_input("Ask a question across all your PDFs")
    if question:
        with st.spinner("Thinking..."):
            docs = retrieve(question)
            reply = answer(question, docs)

        st.markdown(f"**A:** {reply}")

        # Unique file + page pairs, in retrieval order
        sources = list(dict.fromkeys(f"{d.metadata['source']} - page {d.metadata['page']}" for d in docs))
        st.caption("Sources: " + " | ".join(sources))
else:
    st.info("Upload one or more PDFs to get started.")