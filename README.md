# hacklytics2026
Hacklytics

how to use the RAG:
# Install dependencies (or use .venv-rag)
pip install -r requirements-rag.txt

# Build index (run once)
cd data-processing && python rag_build.py

# Query
python rag_query.py "Which patients have VSD and DORV?"
python rag_query.py "Patients with severe dilation" --no-llm -k 5   # retrieval only
python rag_query.py -i   # interactive mode