# Enhanced Legal RAG System

An advanced Retrieval-Augmented Generation (RAG) system specifically designed for legal document analysis and question answering, with support for Kazakhstani legislation.

## 🚀 Features

### Core RAG Capabilities
- **Hybrid Search**: Combines dense vector search with sparse keyword matching
- **Cross-Encoder Re-ranking**: Uses advanced re-ranking models for better result quality
- **Semantic Chunking**: Intelligent text segmentation based on sentence boundaries
- **Conversation Memory**: Maintains context across multiple interactions
- **Enhanced Context Building**: Smart context assembly with token limits

### Technical Features
- **Multiple Search Modes**: Simple, dense-only, and hybrid search options
- **Real-time Evaluation**: Built-in performance monitoring and metrics
- **RESTful API**: FastAPI-based web service with comprehensive endpoints
- **Interactive CLI**: Command-line interface for testing and exploration
- **Error Handling**: Robust error handling and fallback mechanisms

## 📁 Project Structure

```
practice1/
├── data/
│   ├── raw/                    # Original legal documents
│   └── chunks/                 # Processed document chunks
├── main.py                     # FastAPI web service
├── rag_system.py              # Core RAG system implementation
├── preprocess.py              # Document preprocessing and chunking
├── embed_and_index_openai.py  # Embedding generation and indexing
├── query.py                   # Interactive query interface
├── evaluate_rag.py            # System evaluation and testing
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd practice1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file with:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_index_name
   PINECONE_ENVIRONMENT=us-east-1
   ```

## 🔧 Usage

### 1. Document Preprocessing

First, process your legal documents:

```bash
python preprocess.py
```

This will:
- Clean and normalize text
- Split documents into semantic chunks
- Save chunks with metadata

### 2. Create Vector Index

Generate embeddings and create the search index:

```bash
python embed_and_index_openai.py
```

This will:
- Generate embeddings using OpenAI
- Create local embeddings for hybrid search
- Upload vectors to Pinecone
- Save indexing statistics

### 3. Start the Web Service

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 4. Interactive Querying

Use the command-line interface:

```bash
python query.py
```

Or query directly:

```bash
python query.py "Какие права имеет работник при увольнении?"
```

### 5. System Evaluation

Test the system performance:

```bash
python evaluate_rag.py
```

## 🌐 API Endpoints

### Search Endpoints

- `POST /search` - Enhanced search with hybrid search and re-ranking
- `POST /search/simple` - Simple search without advanced features

### Conversation Management

- `GET /conversation/{user_id}` - Get conversation history
- `DELETE /conversation/{user_id}` - Clear conversation history

### System Information

- `GET /` - System information and features
- `GET /stats` - System statistics
- `GET /health` - Health check

### Example API Usage

```python
import requests

# Enhanced search
response = requests.post("http://localhost:8000/search", json={
    "query": "Какие права имеет работник при увольнении?",
    "use_hybrid_search": True,
    "use_reranking": True
})

# Simple search
response = requests.post("http://localhost:8000/search/simple", json={
    "query": "Как оформить трудовой договор?"
})

# Get system stats
response = requests.get("http://localhost:8000/stats")
```

## 🔍 Search Modes

### 1. Simple Search
- Basic dense vector search
- No re-ranking
- Fastest response time

### 2. Dense Search with Re-ranking
- Dense vector search
- Cross-encoder re-ranking
- Better result quality

### 3. Hybrid Search
- Combines dense and sparse retrieval
- Cross-encoder re-ranking
- Best overall performance

## 📊 Performance Metrics

The system tracks various performance metrics:

- **Response Time**: Average time to generate responses
- **Success Rate**: Percentage of successful queries
- **Answer Quality**: Length and relevance of answers
- **Source Diversity**: Number of unique sources used
- **Context Utilization**: How much context is used

## 🧪 Evaluation

The evaluation script tests:

- Different search configurations
- Response quality and speed
- Conversation memory functionality
- System statistics and health

Results are saved to `evaluation_results.json` for analysis.

## 🔧 Configuration

### Search Parameters

You can adjust various parameters in `rag_system.py`:

```python
# Search parameters
self.top_k_initial = 20      # Initial search results
self.top_k_final = 5         # Final results after re-ranking
self.rerank_threshold = 0.5  # Re-ranking score threshold

# Conversation memory
self.max_history_length = 10  # Maximum conversation turns
```

### Chunking Parameters

Adjust in `preprocess.py`:

```python
MAX_CHUNK_SIZE = 1000  # Maximum tokens per chunk
MIN_CHUNK_SIZE = 50    # Minimum tokens per chunk
OVERLAP_SIZE = 100     # Overlap between chunks
```

## 🚨 Error Handling

The system includes comprehensive error handling:

- **API Failures**: Graceful fallbacks for OpenAI/Pinecone errors
- **Empty Results**: Informative messages when no relevant documents found
- **Invalid Inputs**: Input validation and sanitization
- **Rate Limiting**: Automatic retry logic for API limits

## 🔒 Security Considerations

- API keys are stored in environment variables
- Input sanitization prevents injection attacks
- Rate limiting prevents abuse
- Error messages don't expose sensitive information

## 📈 Monitoring

The system provides several monitoring capabilities:

- Real-time performance metrics
- Conversation history tracking
- System health checks
- Detailed error logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information
4. Include error logs and system information

## 🔄 Version History

- **v2.0.0**: Enhanced RAG with hybrid search, re-ranking, and conversation memory
- **v1.0.0**: Basic RAG implementation with Pinecone and OpenAI

---

**Note**: This system is designed for educational and research purposes. For production use, ensure proper security measures and compliance with relevant regulations. 