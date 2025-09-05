# build_index.py
import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# Try faiss; fallback to sklearn if unavailable
try:
    import faiss
    FAISS_AVAILABLE = True
except:
    from sklearn.neighbors import NearestNeighbors
    FAISS_AVAILABLE = False

CORPUS_CSV = "sample_corpus.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "models/faiss.index"
META_FILE = "models/meta.json"
EMB_FILE = "models/embeddings.npy"

os.makedirs("models", exist_ok=True)

def load_corpus(path):
    df = pd.read_csv(path)
    # expected columns: id, text, filename (optional), tags (optional), source (optional)
    return df

def build_embeddings(texts, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def save_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, INDEX_FILE)
    np.save(EMB_FILE, embeddings)
    return True

def save_sklearn_index(embeddings):
    nn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(embeddings)
    # sklearn object cannot be saved easily without joblib; save embeddings for fallback
    np.save(EMB_FILE, embeddings)
    return True

def main():
    df = load_corpus(CORPUS_CSV)
    texts = df['text'].fillna('').astype(str).tolist()
    embeddings = build_embeddings(texts)
    if FAISS_AVAILABLE:
        save_faiss_index(embeddings)
        print("Saved FAISS index to", INDEX_FILE)
    else:
        save_sklearn_index(embeddings)
        print("FAISS not available, saved embeddings to", EMB_FILE)
    # Save metadata
    meta = df.to_dict(orient='records')
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved meta to", META_FILE)

if __name__ == "__main__":
    main()
