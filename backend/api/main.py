from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import chromadb
import openai
import pymongo
import networkx as nx
import os

# --- Config ---
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
MONGO_URI = "mongodb://localhost:27017/"
CHROMA_COLLECTION = "legal_docs"

# --- App & Middleware ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MongoDB ---
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client["legally"]
users_col = db["users"]
history_col = db["history"]

# --- ChromaDB ---
chroma_client = chromadb.Client()
if CHROMA_COLLECTION not in [c.name for c in chroma_client.list_collections()]:
    chroma_client.create_collection(CHROMA_COLLECTION)
chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION)

# --- Auth ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = users_col.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return user

# --- Models ---
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    reliability: float
    sources: Optional[List[str]] = None

class AnalyzeRequest(BaseModel):
    file_id: str

class AnalyzeResponse(BaseModel):
    result: str
    reliability: float
    graph: Optional[dict] = None

# --- Auth Endpoints ---
@app.post("/auth/register", response_model=Token)
def register(user: UserCreate):
    if users_col.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_pw = get_password_hash(user.password)
    users_col.insert_one({"username": user.username, "hashed_password": hashed_pw})
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_col.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

# --- Chat (RAG) Endpoint ---
@app.post("/chat/consult", response_model=ChatResponse)
def chat_consult(req: ChatRequest, user=Depends(get_current_user)):
    # 1. Retrieve relevant docs from ChromaDB
    results = chroma_collection.query(query_texts=[req.message], n_results=3)
    sources = [doc["text"] for doc in results["documents"][0]]
    # 2. Compose prompt for OpenAI
    prompt = f"You are a legal assistant. Use the following articles to answer the question.\n\n"
    for i, src in enumerate(sources):
        prompt += f"[Article {i+1}]: {src}\n"
    prompt += f"\nQuestion: {req.message}\nAnswer:"
    # 3. Call OpenAI (replace with your API key)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=512
    )
    answer = response.choices[0].message.content.strip()
    # 4. Reliability: use average similarity score (mocked here)
    reliability = float(results["distances"][0][0]) if results["distances"][0] else 0.8
    return {"answer": answer, "reliability": reliability, "sources": sources}

# --- Document Analysis Endpoint ---
@app.post("/document/analyze", response_model=AnalyzeResponse)
def analyze_document(file: UploadFile = File(...), user=Depends(get_current_user)):
    # 1. Read file and chunk (mocked)
    text = file.file.read().decode("utf-8")
    # 2. Analyze with RAG (mocked)
    result = "Document complies with legislation."
    reliability = 0.92
    # 3. Generate graph (mocked)
    G = nx.Graph()
    G.add_node("Document")
    G.add_node("Law")
    G.add_edge("Document", "Law")
    graph_data = nx.node_link_data(G)
    return {"result": result, "reliability": reliability, "graph": graph_data}

# --- History Endpoints ---
@app.get("/history")
def get_history(user=Depends(get_current_user)):
    history = list(history_col.find({"username": user["username"]}, {"_id": 0}))
    return {"history": history}

@app.post("/history")
def save_history(entry: dict, user=Depends(get_current_user)):
    entry["username"] = user["username"]
    entry["timestamp"] = datetime.utcnow().isoformat()
    history_col.insert_one(entry)
    return {"status": "ok"}

# --- Graph Endpoint ---
@app.get("/graph")
def get_graph():
    # Return a sample graph (for visualization)
    G = nx.Graph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B")
    return nx.node_link_data(G) 