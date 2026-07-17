"""
Streamlit chat UI for the RAG Document Assistant.
Renders the assistant's markdown answer directly. Note: the backend
returns image paths as relative URLs (/static/images/xyz.png) since it
doesn't know its own public address. The frontend must rewrite these to
absolute URLs pointing at the backend host before rendering, otherwise
the browser tries to load them relative to Streamlit's own origin and
they silently fail.
"""
import re

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000/api/v1"
BACKEND_ROOT = "http://localhost:8000"  # static files are mounted here, not under /api/v1

st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "selected_document_id" not in st.session_state:
    st.session_state.selected_document_id = None


def fetch_documents():
    try:
        resp = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        resp.raise_for_status()
        st.session_state.documents = resp.json()
    except requests.RequestException as exc:
        st.sidebar.error(f"Could not reach backend: {exc}")


def upload_document(file):
    with st.spinner(f"Processing {file.name}... this can take a minute for scanned pages."):
        files = {"file": (file.name, file.getvalue())}
        resp = requests.post(f"{API_BASE_URL}/documents/upload", files=files, timeout=600)
        if resp.status_code == 200:
            data = resp.json()
            if data["status"] == "ready":
                st.sidebar.success(f"✅ {file.name} processed")
            else:
                st.sidebar.error(f"❌ Processing failed for {file.name}")
            fetch_documents()
        else:
            st.sidebar.error(f"Upload failed: {resp.text}")


def make_image_urls_absolute(markdown_text: str) -> str:
    """
    Converts ![caption](/static/images/xyz.png) into
    ![caption](http://localhost:8000/static/images/xyz.png) so the
    browser can actually resolve and load the image.
    """
    pattern = re.compile(r"!\[([^\]]*)\]\((/static/[^)]+)\)")
    return pattern.sub(lambda m: f"![{m.group(1)}]({BACKEND_ROOT}{m.group(2)})", markdown_text)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("📄 Document Assistant")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "pptx", "png", "jpg", "jpeg"],
    )
    if uploaded_file is not None and st.button("Upload & Process"):
        upload_document(uploaded_file)

    st.divider()
    st.subheader("Your Documents")
    fetch_documents()

    if st.session_state.documents:
        doc_options = {"All documents": None}
        for doc in st.session_state.documents:
            status_icon = "✅" if doc["status"] == "ready" else "⏳" if doc["status"] == "processing" else "❌"
            doc_options[f"{status_icon} {doc['filename']}"] = doc["document_id"]

        selected_label = st.selectbox("Search scope", list(doc_options.keys()))
        st.session_state.selected_document_id = doc_options[selected_label]
    else:
        st.caption("No documents uploaded yet.")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat ----------------
st.title("Ask your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about your uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/query",
                    json={
                        "question": question,
                        "document_id": st.session_state.selected_document_id,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = make_image_urls_absolute(data["answer"])
                st.markdown(answer)

                if data["source_chunks"]:
                    with st.expander(f"📚 Sources ({len(data['source_chunks'])} excerpts)"):
                        for c in data["source_chunks"]:
                            page = f"page {c['page_number']}" if c["page_number"] else "n/a"
                            st.caption(f"**{page}** · relevance {c['relevance_score']}")
                            st.text(c["content"][:300] + ("..." if len(c["content"]) > 300 else ""))

            except requests.RequestException as exc:
                answer = f"⚠️ Error reaching backend: {exc}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})