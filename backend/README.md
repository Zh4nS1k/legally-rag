# Legally Backend

## Setup

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set environment variables:**

- `OPENAI_API_KEY` (for OpenAI RAG)
- (Optional) Edit `MONGO_URI` in `api/main.py` if your MongoDB is not local.

3. **Run the server:**

```bash
uvicorn api.main:app --reload
```

- The API will be available at `http://localhost:8000`

## Endpoints

- `POST /auth/register` — Register new user
- `POST /auth/login` — Login, get JWT
- `POST /chat/consult` — Legal AI chat (RAG)
- `POST /document/analyze` — Document compliance check
- `GET/POST /history` — Get/save user history
- `GET /graph` — Get sample graph for visualization

## Notes

- Requires MongoDB running locally by default.
- Requires ChromaDB (runs in-process by default).
- Requires OpenAI API key for chat/analysis.
