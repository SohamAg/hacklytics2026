# Heart Features RAG

Retrieval-Augmented Generation (RAG) on `heart_features.csv` — semantic search over cardiac CMR patient data with optional LLM answers.

## Setup

```bash
pip install -r ../requirements-rag.txt
# Or use the venv: source ../.venv-rag/bin/activate
```

## Build Index (run once)

```bash
python rag_build.py
```

Creates `heart_rag_index.pkl` and `heart_rag_index.npz`. Embeds 59 patient records using `all-MiniLM-L6-v2`.

## Query

```bash
# One-off question
python rag_query.py "Which patients have VSD and DORV?"

# Retrieval only (no LLM)
python rag_query.py "Patients with severe dilation" --no-llm -k 5

# Interactive mode
python rag_query.py -i
```

## LLM Options

- **Ollama** (local): `pip install ollama` and run `ollama run llama3.2`
- **OpenAI**: set `OPENAI_API_KEY`

Without either, use `--no-llm` to get retrieved context only.

## Indexed Content

Each patient is encoded as text with: demographics (age, category), chamber volumes (LV, RV, LA, RA, Aorta, PA, SVC, IVC), total heart volume, LV/RV and LA/RA ratios, and conditions (VSD, ASD, DORV, Fontan, etc.).
