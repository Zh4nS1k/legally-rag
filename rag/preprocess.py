import os
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

RAW_DIR = "data/raw"
CHUNK_DIR = "data/chunks"
MAX_CHUNK_SIZE = 1000  # tokens
MIN_CHUNK_SIZE = 50    # tokens
OVERLAP_SIZE = 100     # tokens

os.makedirs(CHUNK_DIR, exist_ok=True)

# Initialize sentence model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def estimate_tokens(text: str) -> int:
    """Simple token estimation (rough approximation)"""
    return int(len(text.split()) * 1.3)

def semantic_chunk_text(text: str) -> List[Dict[str, str]]:
    """Create semantic chunks based on sentence boundaries and token count"""
    text = clean_text(text)
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        # If adding this sentence would exceed max size, save current chunk
        if current_tokens + sentence_tokens > MAX_CHUNK_SIZE and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if estimate_tokens(chunk_text) >= MIN_CHUNK_SIZE:
                chunks.append({
                    'text': chunk_text,
                    'tokens': estimate_tokens(chunk_text)
                })
            
            # Start new chunk with overlap
            overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_tokens = sum(estimate_tokens(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if estimate_tokens(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append({
                'text': chunk_text,
                'tokens': estimate_tokens(chunk_text)
            })
    
    return chunks

def process_files():
    """Process all files in the raw directory"""
    print("🔄 Starting semantic chunking process...")
    
    for filename in tqdm(os.listdir(RAW_DIR), desc="Processing files"):
        if not filename.endswith(".txt"):
            continue
            
        file_path = os.path.join(RAW_DIR, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Create semantic chunks
            chunks = semantic_chunk_text(text)
            
            # Save chunks
            base_name = filename.replace(".txt", "")
            for i, chunk in enumerate(chunks):
                chunk_filename = f"{base_name}_chunk_{i+1}.txt"
                chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
                
                with open(chunk_path, "w", encoding="utf-8") as cf:
                    cf.write(chunk['text'])
                
                # Also save metadata
                metadata_filename = f"{base_name}_chunk_{i+1}_meta.txt"
                metadata_path = os.path.join(CHUNK_DIR, metadata_filename)
                
                with open(metadata_path, "w", encoding="utf-8") as mf:
                    mf.write(f"tokens:{chunk['tokens']}\n")
                    mf.write(f"source:{filename}\n")
                    mf.write(f"chunk_id:{i+1}\n")
        
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    print(f"✅ Semantic chunking completed! Processed {len([f for f in os.listdir(RAW_DIR) if f.endswith('.txt')])} files")

if __name__ == "__main__":
    process_files()
