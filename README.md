# Corrective RAG System

Implementation of Corrective Retrieval-Augmented Generation (CRAG) with LangChain.

## What is Corrective RAG?

Corrective RAG evaluates retrieved documents and takes corrective actions:

- **Correct Path** (score >= 0.5): Use filtered documents
- **Incorrect Path** (score <= 0.3): Web search fallback
- **Ambiguous Path** (0.3-0.5): Hybrid approach

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create `.env` file:

```
AVALAI_API_KEY=your-api-key
AVALAI_BASE_URL=https://api.avalai.ir/v1

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=corrective-rag
```

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501 and upload a PDF to chat with it.

## Project Structure

```
src/
  vector_store.py      # ChromaDB storage
  evaluator.py         # Relevance scoring (cross-encoder)
  corrective_paths.py  # Correct/Incorrect/Ambiguous paths
  corrective_rag.py    # Main orchestrator
app.py                 # Streamlit chat interface
```

## Technologies

- LangChain
- ChromaDB
- Aval AI (embeddings + LLM)
- HuggingFace cross-encoder
- Streamlit
