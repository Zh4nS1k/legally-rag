import os
import json
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

# === Шаг 1: Загрузка ключей ===
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "legally-index"

# === Шаг 2: Настройка клиентов ===
openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize sentence transformer for local embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# === Шаг 3: Создание индекса, если не существует ===
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    print(f"📎 Индекс '{INDEX_NAME}' не найден. Создаём новый...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
    )
else:
    print(f"✅ Индекс '{INDEX_NAME}' уже существует.")

index = pc.Index(INDEX_NAME)

# === Шаг 4: Загрузка текстов чанков с метаданными ===
chunk_dir = "data/chunks"
texts = []
metadatas = []

print("📖 Загрузка чанков...")

for filename in os.listdir(chunk_dir):
    if not filename.endswith(".txt") or filename.endswith("_meta.txt"):
        continue
        
    path = os.path.join(chunk_dir, filename)
    meta_path = os.path.join(chunk_dir, filename.replace(".txt", "_meta.txt"))
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        if len(text) < 10:
            continue
            
        # Load metadata if available
        metadata = {"filename": filename, "text": text[:200] + "..." if len(text) > 200 else text}
        
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as mf:
                for line in mf:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        metadata[key] = value
        
        texts.append(text)
        metadatas.append(metadata)
        
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        continue

print(f"📊 Загружено {len(texts)} чанков")

# === Шаг 5: Функция получения эмбеддингов ===
def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding from OpenAI"""
    try:
        text = text.replace("\n", " ")
        res = openai.embeddings.create(input=[text], model=model)
        return res.data[0].embedding
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return None

def get_local_embedding(text):
    """Get embedding using sentence transformers"""
    try:
        embedding = sentence_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"❌ Error getting local embedding: {e}")
        return None

# === Шаг 6: Векторизация и загрузка в Pinecone ===
batch_size = 50  # Reduced batch size for better error handling

print("🚀 Начинаем индексацию...")

successful_uploads = 0
failed_uploads = 0

for i in tqdm(range(0, len(texts), batch_size), desc="📦 Индексация в Pinecone"):
    batch_texts = texts[i:i + batch_size]
    batch_metadatas = metadatas[i:i + batch_size]
    
    vectors_to_upsert = []
    
    for j, (text, metadata) in enumerate(zip(batch_texts, batch_metadatas)):
        # Get OpenAI embedding
        embedding = get_embedding(text)
        
        if embedding is None:
            failed_uploads += 1
            continue
            
        # Get local embedding for hybrid search
        local_embedding = get_local_embedding(text)
        
        # Enhanced metadata
        enhanced_metadata = {
            **metadata,
            "text_length": len(text),
            "local_embedding": local_embedding if local_embedding else [],
            "embedding_model": "text-embedding-3-small"
        }
        
        vectors_to_upsert.append({
            "id": f"doc-{i + j}",
            "values": embedding,
            "metadata": enhanced_metadata
        })
    
    # Upload batch
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            successful_uploads += len(vectors_to_upsert)
        except Exception as e:
            print(f"❌ Error uploading batch {i//batch_size}: {e}")
            failed_uploads += len(vectors_to_upsert)

print(f"✅ Индексация завершена!")
print(f"📊 Успешно загружено: {successful_uploads}")
print(f"❌ Ошибок: {failed_uploads}")

# Save index statistics
stats = {
    "total_chunks": len(texts),
    "successful_uploads": successful_uploads,
    "failed_uploads": failed_uploads,
    "index_name": INDEX_NAME
}

with open("index_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("📈 Статистика сохранена в index_stats.json")
