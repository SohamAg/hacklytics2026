#!/usr/bin/env python3
"""Query the heart_features RAG. Retrieves relevant patients and optionally generates answers."""

import pickle
import numpy as np
from pathlib import Path


def query_rag(
    question: str,
    index_path: str = "heart_rag_index.pkl",
    top_k: int = 5,
    use_llm: bool = True,
):
    """Retrieve relevant patient data and optionally generate an answer."""
    from sentence_transformers import SentenceTransformer

    index_path = Path(index_path)
    npz_path = index_path.with_suffix(".npz")

    if not index_path.exists() or not npz_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. Run: python rag_build.py"
        )

    # Load index
    with open(index_path, "rb") as f:
        index_data = pickle.load(f)
    data = np.load(npz_path)
    embeddings = data["embeddings"]

    documents = index_data["documents"]
    metadatas = index_data["metadatas"]

    # Load same embedding model (use local cache to avoid network)
    cache_dir = index_path.parent / ".hf_cache"
    model_name = index_data.get("model_name", "all-MiniLM-L6-v2")
    if cache_dir.exists():
        # Load from local cache - try cache_folder first
        model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_dir),
            local_files_only=True,
        )
    else:
        model = SentenceTransformer(model_name)
    query_emb = model.encode([question])[0]
    query_emb = query_emb / (np.linalg.norm(query_emb) or 1)

    # Cosine similarity (embeddings already normalized)
    scores = np.dot(embeddings, query_emb)
    top_indices = np.argsort(scores)[::-1][:top_k]

    docs = [documents[i] for i in top_indices]
    metas = [metadatas[i] for i in top_indices]
    distances = [float(1 - scores[i]) for i in top_indices]  # distance = 1 - similarity

    context = "\n\n".join(
        f"[Patient {m.get('patient', '?')}] {d}"
        for d, m in zip(docs, metas)
    )

    if not context:
        return "No relevant patients found in the index.", []

    retrieved = list(zip(docs, metas, distances))

    if use_llm:
        answer = _generate_with_llm(question, context)
        return answer, retrieved
    else:
        return context, retrieved


def _generate_with_llm(question: str, context: str) -> str:
    """Generate answer using available LLM (Ollama or OpenAI)."""
    prompt = f"Context from heart CMR database:\n\n{context}\n\nQuestion: {question}"
    system = "You are a cardiac imaging assistant. Answer using only the provided patient data. If the data doesn't contain relevant info, say so."

    # Try Ollama first (local, no API key)
    try:
        import ollama
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except ImportError:
        pass
    except Exception as e:
        if "Connection" not in str(e) and "404" not in str(e):
            return f"[Ollama error: {e}]\n\nRetrieved context:\n\n{context}"
        pass

    # Fall back to OpenAI
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return (
            f"[LLM unavailable. Install ollama (pip install ollama) or set OPENAI_API_KEY. Error: {e}]\n\n"
            "Retrieved context (use --no-llm to always show this):\n\n" + context
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query heart_features RAG")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--index", default="heart_rag_index.pkl", help="Path to index pickle")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of patients to retrieve")
    parser.add_argument("--no-llm", action="store_true", help="Only show retrieved context, no LLM")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    index_path = script_dir / args.index

    def run_query(q: str):
        answer, retrieved = query_rag(
            q, index_path=str(index_path), top_k=args.top_k, use_llm=not args.no_llm
        )
        print("\n--- Answer ---")
        print(answer)
        if retrieved and not args.no_llm:
            print("\n--- Retrieved patients ---")
            for doc, meta, dist in retrieved:
                print(f"  • Patient {meta.get('patient', '?')} (category: {meta.get('category', '?')}) [dist: {dist:.3f}]")

    if args.interactive:
        print("Heart features RAG - Interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                q = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit", "q"):
                break
            run_query(q)
            print()
    elif args.question:
        run_query(args.question)
    else:
        parser.print_help()
        print("\nExample: python rag_query.py 'Which patients have VSD and DORV?'")


if __name__ == "__main__":
    main()
