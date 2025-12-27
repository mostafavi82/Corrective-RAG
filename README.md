# Corrective RAG System

Implementation of Corrective Retrieval-Augmented Generation (CRAG) with LangChain.

## What is Corrective RAG?

Corrective RAG evaluates retrieved documents and takes corrective actions:

- **Correct Path** (relevance >= 50%): Use filtered documents
- **Incorrect Path** (relevance <= 30%): Web search fallback with Tavily
- **Ambiguous Path** (30-50%): Hybrid approach (documents + web)

## Architecture

```
Query -> Embed -> Retrieve -> Evaluate (LLM) -> Corrective Action -> Generate
                                   |
                            [Relevance Check]
                                   |
                      +------------+------------+
                      |            |            |
                   CORRECT     AMBIGUOUS    INCORRECT
                   (>=50%)     (30-50%)      (<=30%)
                      |            |            |
                   Filter     Filter+Web    Web Search
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create `.env` file:

```env
# Aval AI (for LLM evaluation and answer generation)
AVALAI_API_KEY=your-avalai-key
AVALAI_BASE_URL=https://api.avalai.ir/v1

# Metis AI (for embeddings)
METIS_API_KEY=your-metis-key
METIS_BASE_URL=https://api.metisai.ir/openai/v1

# Tavily (for web search)
TAVILY_API_KEY=your-tavily-key

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
  vector_store.py       # ChromaDB storage with Metis embeddings
  llm_evaluator.py      # LLM-based relevance evaluation
  corrective_paths.py   # Correct/Incorrect/Ambiguous paths + Tavily search
  corrective_rag.py     # Main orchestrator
  metis_embeddings.py   # Metis AI embeddings wrapper
  avalai_embeddings.py  # Aval AI embeddings wrapper
app.py                  # Streamlit chat interface
```

## Technologies

| Component | Technology | Provider |
|-----------|------------|----------|
| Embeddings | text-embedding-3-small | Metis AI |
| Relevance Evaluation | qwen2.5-vl-3b-instruct | Aval AI |
| Answer Generation | gpt-4o-mini | Aval AI |
| Question Rewriter | gpt-4o-mini | Metis AI |
| Web Search | Tavily Search | Tavily |
| Vector Store | ChromaDB | Local |
| UI | Streamlit | - |

## How It Works

1. **Document Loading**: Upload PDF, extract text, split into chunks
2. **Embedding**: Convert chunks to vectors using Metis AI
3. **Retrieval**: Find similar chunks using ChromaDB
4. **Evaluation**: LLM evaluates relevance of each document (batch)
5. **Path Selection**: Choose action based on relevance ratio
6. **Corrective Action**: Filter docs, web search, or hybrid
7. **Generation**: Generate answer with selected context
