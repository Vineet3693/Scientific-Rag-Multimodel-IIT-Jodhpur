# Future Extensions

This directory contains skeleton code and plans for future extensions to the Scientific Multimodal RAG system.

---

## Directory Structure

```
future/
├── README.md              # This file — overview of future extensions
├── __init__.py            # Package init
├── hybrid/                # Medical + Scientific hybrid RAG
│   ├── __init__.py
│   ├── query_router.py    # Keyword-based query router
│   └── adapter/           # Adapters for medical pipeline
│       └── __init__.py
├── advanced/              # Advanced features
│   └── __init__.py
├── backend/               # REST API backend
│   └── routes/
│       └── __init__.py
├── scaling/               # Scaling and deployment
│   └── __init__.py
└── frontend/              # Custom web frontend
    └── index.html         # Basic HTML skeleton
```

---

## Planned Extensions

### 1. Hybrid RAG (Medical + Scientific)

A query router that classifies incoming queries as medical or scientific and routes them to the appropriate pipeline. This merges the current scientific RAG with Gokul's medical RAG.

- **Status**: Skeleton code in `hybrid/query_router.py`
- **Priority**: High
- **ETA**: Phase 8

### 2. Advanced Retrieval

- Re-ranking with cross-encoders
- Query expansion with LLM
- Multi-hop retrieval for complex questions
- Adaptive retrieval based on query type

### 3. REST API Backend

- FastAPI-based REST API for the RAG pipeline
- WebSocket support for streaming answers
- Authentication and rate limiting
- Batch query endpoints

### 4. Scaling

- Model parallelism for larger models
- Distributed retrieval across multiple GPUs
- Caching layer for frequent queries
- Horizontal scaling with load balancer

### 5. Custom Frontend

- React-based web interface
- Real-time answer streaming
- Interactive source exploration
- PDF viewer with highlighted passages
- Dark/light mode

---

## Contributing

When implementing a future extension:

1. Create a feature branch from `main`
2. Follow the existing code style (English docstrings, type hints)
3. Add tests in `tests/` for new functionality
4. Update `TRACKER.md` with the new file status
5. Submit a pull request with a clear description
