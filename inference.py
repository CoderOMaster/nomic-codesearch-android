"""
Run nomic-codesearch-onnx to embed code and queries, then find the most similar code snippet.

Install deps:
    pip install onnxruntime transformers numpy
"""
import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    session = ort.InferenceSession(os.path.join(MODEL_DIR, "model_int8.onnx"))
    return tokenizer, session


def embed(texts: list[str], tokenizer, session, max_length: int = 512) -> np.ndarray:
    """Return L2-normalised sentence embeddings, shape (len(texts), 768)."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    outputs = session.run(
        ["sentence_embedding"],
        {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        },
    )
    embeddings = outputs[0]  # (batch, 768)
    # L2 normalise so dot-product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def search(query: str, code_snippets: list[str], tokenizer, session, top_k: int = 3):
    """Return the top-k most similar code snippets for a natural-language query."""
    query_emb = embed([query], tokenizer, session)          # (1, 768)
    code_embs = embed(code_snippets, tokenizer, session)    # (n, 768)
    scores = (query_emb @ code_embs.T)[0]                  # (n,)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(code_snippets[i], float(scores[i])) for i in top_indices]


if __name__ == "__main__":
    tokenizer, session = load_model()

    # --- example: embed a single snippet ---
    code = "def add(a, b): return a + b"
    emb = embed([code], tokenizer, session)
    print(f"Embedding shape: {emb.shape}")   # (1, 768)
    print(f"First 5 dims:    {emb[0, :5]}")

    # --- example: code search ---
    snippets = [
        "def add(a, b): return a + b",
        "def binary_search(arr, target): ...",
        "SELECT * FROM users WHERE id = ?",
        "for i in range(10): print(i)",
    ]
    query = "function that adds two numbers"
    results = search(query, snippets, tokenizer, session, top_k=2)
    print(f"\nTop results for: '{query}'")
    for snippet, score in results:
        print(f"  [{score:.4f}] {snippet}")
