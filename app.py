# app.py — Viswam AI Chat Assistant

# -------------------------------
# 1️⃣ Streamlit setup (must be first)
# -------------------------------
import streamlit as st
st.set_page_config(page_title="Viswam AI — Chat Assistant", layout="centered")

# -------------------------------
# 2️⃣ Other imports
# -------------------------------
import json
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------
# 3️⃣ Load models & metadata
# -------------------------------
INDEX_PATH = "models/faiss.index"
META_PATH = "models/meta.json"

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load metadata
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 4️⃣ Streamlit UI
# -------------------------------
st.title("Viswam AI — Chat Assistant")
st.write("Ask questions based on your uploaded corpus!")

query = st.text_input("Enter your question here:")

if query:
    # Generate embedding for query
    query_embedding = model.encode([query])
    # Search top 3 relevant corpus entries
    D, I = index.search(query_embedding, k=3)

    st.subheader("Top relevant answers:")
    for idx in I[0]:
        st.write("- " + meta[idx]["text"])
