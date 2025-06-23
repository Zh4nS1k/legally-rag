from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import os
from pinecone import Pinecone

# === Инициализация ===
load_dotenv()
app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legally-index")

# === Модели запроса ===
class EmbedRequest(BaseModel):
    user_id: str
    text: str
    metadata: dict = {}

class SearchRequest(BaseModel):
    user_id: str
    query: str

# === Вставка в Pinecone ===
@app.post("/embed")
def embed_and_store(data: EmbedRequest):
    try:
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=data.text
        )['data'][0]['embedding']

        index.upsert([
            (
                f"{data.user_id}_{hash(data.text)}",  # уникальный ID
                embedding,
                {
                    **data.metadata,
                    "text": data.text,
                    "user_id": data.user_id
                }
            )
        ])

        return {"message": "Документ успешно сохранён", "vector_dim": len(embedding)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Поиск по вектору и ответ GPT ===
@app.post("/search")
def search_docs(data: SearchRequest):
    try:
        # Получение эмбеддинга для запроса
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=data.query
        )['data'][0]['embedding']

        # Поиск похожих документов в Pinecone
        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True,
            filter={"user_id": {"$eq": data.user_id}}  # фильтрация по пользователю
        )

        matches = results.get('matches', [])
        if not matches:
            return {"matches": None, "answer": "Ничего не найдено по запросу."}

        # Сбор контекста
        context = "\n".join([m['metadata'].get("text", "") for m in matches])

        # Запрос к OpenAI GPT
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Ты — AI-юрист. Отвечай строго по контексту закона."},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {data.query}"}
            ],
            temperature=0.2,
            max_tokens=800
        )

        return {
            "matches": matches,
            "answer": gpt_response['choices'][0]['message']['content']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
