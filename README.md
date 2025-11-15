# 🤖 Henry Bot M2 - Enhanced LLM Agent with RAG

An advanced FAQ LLM agent with **Retrieval-Augmented Generation (RAG)** capabilities, built on the foundation of Henry Bot M1 with enhanced architecture and comprehensive API server.

## 🌟 Key Features

- **🔗 RAG Integration**: Document-based context retrieval with FAISS vector storage
- **🌐 REST API**: FastAPI server with automatic Swagger/ReDoc documentation
- **🔒 Enterprise Security**: API authentication, rate limiting, security headers
- **📊 Real-time Metrics**: Performance tracking, latency monitoring, cost analysis
- **🛡️ Safety Features**: Advanced adversarial prompt detection
- **🧩 Modular Architecture**: Clean separation of concerns with orchestrator pattern
- **📝 Prompt Engineering**: Multiple techniques (few-shot, chain-of-thought)
- **🐳 Docker Ready**: Complete containerization support

## 📋 Report

The implementation is based on Henry Bot M1 with significant enhancements:
- **Architecture**: Modular design with main.py as pure orchestrator 
- **RAG System**: FAISS-based vector storage with Sentence-Transformers embeddings
- **API Server**: RESTful endpoints with comprehensive documentation
- **Performance**: Enhanced metrics tracking and analytics

### RAG System details:
- **Document Upload**: TXT/MD/PDF/DOCX file support
  The system accepts ingestion of text documents in common formats: plain text (.txt), Markdown (.md), PDF (.pdf), and Word (.docx). The pipeline converts these files to text for processing.

- **Chunking strategy**: (Sliding windows) Overlapping text chunks for context retrieval
  A sliding‑window chunking strategy is used to divide documents into overlapping segments. This method slides a fixed‑size window across the text; each window overlaps with the previous segment to preserve context. This improves retrieval relevance by ensuring that important information spanning chunk boundaries is captured.

  The cost is redundancy (more storage and processing), but sliding windows are especially useful for unstructured text such as conversations or transcripts where important context may span multiple sentences or paragraphs.

- **Embeddings**: Sentence-Transformers for semantic representation
  Each fragment is transformed into a vector using a pre‑trained Sentence Transformers model (all‑MiniLM‑L6‑v2). This model converts text into a high‑dimensional vector space so that semantically similar texts are close together. This enables effective similarity search based on meaning rather than just keywords.

  At query time, the question is also embedded into the same vector space and compared to the fragment embeddings to find those with the highest semantic similarity.

- **Vector Storage**: FAISS for efficient similarity search
  The embeddings are stored in a FAISS (Facebook AI Similarity Search) index for efficient similarity search. FAISS is designed for nearest‑neighbour search across large volumes of vectors, which makes it suitable for RAG systems.

  The vector store allows retrieval of the k nearest vectors; when queried, it quickly returns the k most similar fragments.

- **Context Retrieval**: Top-K similar chunks with relevance scoring
  For each query, the system retrieves the k most relevant fragments. One example RAG pipeline retrieves the top three fragments using a FAISS vector store to answer a question.

  Retrieving multiple overlapping fragments ensures that the context provided covers different sections of the document and reduces omissions. 

- **Prompt Augmentation**: Contextual prompts for LLM queries
  The retrieved fragments are injected into the large language model’s prompt. A common template instructs the model to answer based only on the context and to say when there is not enough information .

  This prompt augmentation grounds the model’s response in factual information from the document and discourages hallucination.

- **RAG-Augmented Responses**: LLM prompts enriched with retrieved context
  By combining semantic retrieval with prompt augmentation, the system produces responses enriched with contextual information. The model’s answer is based on retrieved facts rather than solely on its pre‑trained knowledge.

  Augmented responses improve accuracy and reduce hallucination because the model has access to relevant supporting information during generation.

All the functionalities are encapsulated in modular components for easy maintenance and extension.

---

## 🚀 Server Quick Start

### Prerequisites
- Python 3.9+
- OpenRouter API key (free tier available)

### Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/estebmaister/henry_bot_M2.git
cd henry_bot_M2

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys (see below)

# 5. Start the server
python -m src.main server
# Test the server comfortably from: http://0.0.0.0:8000/swagger

## IMPORTANT: For the first time running the chat endpoint, it will download the embedding model (~90MB)
## and cache it locally in `~/.cache/torch/sentence_transformers/` for future runs.
```

### Environment Configuration

Create a `.env` file with your settings:

```bash
# Required: Get from https://openrouter.ai/settings/keys
OPENROUTER_API_KEY=your-openrouter-api-key-here

# Required: Your secret API key for client authentication
API_KEY=henry_bot_8285994a7534ff6bfc6db2698ce8203d
## This key must be included in request headers as `X-API-Key`
## Is currently easy to get from public endpoints for testing purposes.
## In the future it will be generated per user.

# Optional: Model selection (default: free Gemini model)
MODEL_NAME=google/gemini-2.0-flash-exp:free

# Optional: Server configuration
HOST=0.0.0.0
PORT=8000
```

## 🚀 Features Overview

### Core Capabilities
- **JSON-formatted responses** for seamless integration
- **Multiple prompt engineering techniques** for improved output quality
- **Comprehensive metrics tracking**:
  - ⏱️ *Response latency* (ms)
  - 🪙 *Token usage* (prompt + completion)
  - 💰 *API cost estimation*
  - 🎯 *RAG retrieval scores*

### Enhanced Features (M2)
- **RAG System**: Document retrieval with similarity scoring
- **API Server**: RESTful endpoints with automatic documentation
- **Authentication**: API key-based security with rate limiting (60/minute)
- **Multiple Prompting Techniques**: simple, few_shot, chain_of_thought
- **Comprehensive Metrics**: Latency, tokens, costs, performance tracking
- **Error Handling**: Robust error handling with structured responses
- **Security**: Rate limiting, CORS, security headers, input validation

### 🔄 RAG System
- **Document Upload**: Support for TXT, MD, PDF, DOCX files
- **Vector Storage**: FAISS-based similarity search
- **Context Retrieval**: RAG-augmented responses with scoring
- **Semantic Search**: Advanced document matching
- **Prompt Augmentation**: Contextual prompts for LLM queries
- **Configurable Parameters**: Top-K retrieval, scoring thresholds

### 🛡️ Safety & Security
- **Adversarial Detection**: Multiple pattern types for prompt injection
- **API Authentication**: Secure key-based access control
- **Rate Limiting**: Prevent abuse with configurable limits
- **Input Validation**: Comprehensive request validation
- **Security Headers**: CSRF, XSS protection headers

---

## 🏗️ Architecture

```
src/
├── main.py                    # 🎯 Module orchestrator ONLY
├── core/                      # ⚙️ Configuration & agent logic
│   ├── config.py             # Environment-based settings
│   ├── agent.py              # Main HenryBot orchestrator
│   └── exceptions.py         # Custom exception types
├── modules/                   # 📦 Functional modules
│   ├── prompting/            # 💬 Enhanced from M1 + RAG support
│   │   ├── engine.py         # RAG-aware prompt engineering
│   │   ├── safety.py         # Enhanced adversarial detection
│   │   └── templates/        # System prompt templates
│   ├── metrics/              # 📊 Enhanced from M1 + RAG tracking
│   │   └── tracker.py        # Performance metrics with RAG scores
│   ├── rag/                  # 🔄 Document retrieval system
│   │   ├── retriever.py      # FAISS-based similarity search
│   │   ├── chunking.py       # Document chunking strategies
│   │   ├── embeddings.py     # Embedding generation service
│   │   ├── processor.py      # Document processing pipeline
│   │   ├── storage.py        # Vector & document storage interfaces
│   │   └── chunk_store.py    # JSON-based chunk metadata storage
│
│   ├── api/                  # 🌐 FastAPI server
│   │   ├── server.py         # Application with routes
│   │   ├── schemas.py        # Pydantic models
│   │   └── middleware.py     # Auth, rate limiting, logging
│   └── logging/              # 📝 Enhanced from M1 + RAG analytics
│       └── logger.py         # Structured logging with CSV export
└── utils/                    # 🔧 Helper functions
└── data/                     # 📁 Static data files
└── logs/                     # 📁 Log files and metrics
```

---

## 🛠️ Tech Stack

### Core Technologies
- **Language**: Python 3.10+
- **LLM API**: [OpenRouter API](https://openrouter.ai/settings/keys)
- **RAG System**: FAISS + Sentence-Transformers
- **API Framework**: FastAPI with automatic OpenAPI/Swagger
- **Configuration**: Pydantic with environment variables

### Dependencies
- **fastapi**>=0.104.1 - API framework
- **uvicorn**>=0.24.0 - ASGI server
- **openai**>=1.12.0 - LLM API client
- **sentence-transformers**>=2.2.2 - Text embeddings
- **faiss-cpu**>=1.7.4 - Vector similarity search
- **python-dotenv**>=1.0.0 - Environment configuration

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- OpenRouter API key (free tier available)

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/estebmaister/henry_bot_M2.git
cd henry_bot_M2

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your configuration:
# - OPENROUTER_API_KEY="your-api-key-here"
# - API_KEY="your-secret-api-key-here"
# - MODEL_NAME="google/gemini-2.0-flash-exp:free"
```

### Environment Configuration

Create a `.env` file with your settings:

```bash
# API Configuration
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=google/gemini-2.0-flash-exp:free

# Server Configuration
API_KEY=your-secret-api-key-here
HOST=0.0.0.0
PORT=8000

# LLM Parameters
TEMPERATURE=0.7
MAX_TOKENS=500
PROMPTING_TECHNIQUE=few_shot

# RAG Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_TOP_K=3
RESPONSE_SCORING_THRESHOLD=0.7

# ...
```

---

## 🛠️ API Usage

### Starting the Server

```bash
# Start the production server
python -m src.main server

# Server will be available at:
# API: http://0.0.0.0:8000
# Swagger UI: http://0.0.0.0:8000/docs
# ReDoc: http://0.0.0.0:8000/redoc
# Health Check: http://0.0.0.0:8000/health
```

### API Endpoints

#### 1. Health Check (Public)
```bash
curl http://localhost:8000/health
```

#### 2. Chat Endpoint (Protected)
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is pandora experience?",
    "prompt_technique": "few_shot",
    "use_rag": false
  }'
```

**Response:**
```json
{
  "answer": "...",
  "reasoning": null,
  "metrics": {
    "latency_ms": 1234,
    "tokens_total": 67,
    "cost_usd": 0.0001
  },
  "rag": {...},
  "timestamp": "2025-11-11T22:07:47.065972"
}
```

#### 3. Document Upload (Phase 2)
```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
  -H "X-API-Key: your-secret-api-key-here" \
  -F "files=@document.txt"
```

### Available Prompt Techniques

- **`simple`** - Basic question-answer format
- **`few_shot`** - Examples included in prompt (default)
- **`chain_of_thought`** - Step-by-step reasoning

### CLI Interface

```bash
# Test single question
python -m src.main cli "What is pandora experience?"

# Check system status
python -m src.main status
```

## 🧪 Testing & Quality Assurance

### Manual Testing Examples

```bash
# Test authentication (should fail)
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
# Returns: 401 Unauthorized

# Test with different prompting techniques
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain photosynthesis", "prompt_technique": "chain_of_thought"}'
```

---

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Set environment variables
export OPENROUTER_API_KEY="your-api-key-here"
export API_KEY="your-secret-api-key-here"

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Using Docker Directly

```bash
# Build the image
docker build -t henry-bot-m2 .

# Run the container
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY="your-api-key-here" \
  -e API_KEY="your-secret-api-key-here" \
  henry-bot-m2
```

## 📊 Monitoring & Metrics

### Performance Metrics
The system automatically tracks:
- **Response Latency**: API call response times
- **Token Usage**: Prompt and completion tokens
- **Cost Analysis**: API cost estimation
- **Error Tracking**: Comprehensive error logging
- **RAG Metrics**: Context usage and retrieval scores (Phase 2)

### Log Files
- `logs/app.log` - Application logs
- `logs/metrics.csv` - Performance metrics
- `logs/api_requests.log` - API request tracking

### Health Monitoring
```bash
# Check system health
curl http://localhost:8000/health

# Monitor real-time metrics
tail -f logs/metrics.csv
```

---

## 📈 Development Status

### ✅ Phase 1: Core API (Complete)
- [x] Modular architecture with clean separation
- [x] FastAPI server with authentication
- [x] Multiple prompting techniques
- [x] Comprehensive metrics tracking
- [x] Docker deployment ready
- [x] Manual testing infrastructure

### ✅ Phase 2: RAG System
- [x] Complete FAISS-based semantic search implementation
- [x] Async document processing pipeline
- [x] FAISS vector store with similarity scoring
- [x] Sentence-Transformers embeddings generation
- [x] Context retrieval with rich formatting
- [x] Document upload and background processing
- [x] Production-ready RAG system integration

### ⏳ Phase 3: Advanced Features (Planned)
- [ ] Testing & QA enhancements
- [ ] Streaming responses
- [ ] Conversation memory
- [ ] Multi-modal support

---

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. API Returns 500 Error
**Problem**: Internal server error on chat endpoint
**Solution**: Remove `response_format={"type": "json_object"}` from `src/core/agent.py:129`
**Cause**: Google Gemini models don't support structured output format

#### 2. Rate Limiting Errors
**Problem**: "Provider returned error: 429"
**Solution**: Add your own OpenRouter API key or wait for rate limit reset
**Cause**: Free models have usage limits

#### 3. Authentication Failures
**Problem**: 401 Unauthorized errors
**Solution**: Verify API key in `.env` and request headers
**Check**: Ensure `X-API-Key` header is properly set

#### 4. Module Import Errors
**Problem**: Python import errors
**Solution**: Ensure virtual environment is activated and dependencies installed
**Command**: `source venv/bin/activate && pip install -r requirements.txt`

### Getting Help
- **API Reference**: Visit `/docs` endpoint for interactive documentation
- **Logs**: Check `logs/app.log` for detailed error information

---

## 📚 Additional Resources

### Documentation
- **Interactive API**: http://0.0.0.0:8000/docs - Swagger UI
- **ReDoc Documentation**: http://0.0.0.0:8000/redoc - Alternative API docs
- **OpenAPI Spec**: http://0.0.0.0:8000/openapi.json - Machine-readable spec

### External Links
- **OpenRouter API**: https://openrouter.ai/settings/keys - Get your API key
- **FastAPI Documentation**: https://fastapi.tiangolo.com/ - API framework
- **Docker Hub**: https://hub.docker.com/ - Container registry

---

## 👤 Author & Support

Developed by [Esteban Camargo](https://github.com/estebmaister)

📧 **Email**: [estebmaister@gmail.com](mailto:estebmaister@gmail.com)
🌐 **LinkedIn**: [https://linkedin.com/in/estebmaister](https://linkedin.com/in/estebmaister)
🐙 **GitHub**: [https://github.com/estebmaister](https://github.com/estebmaister)

### Project Status
- ✅ **Production Ready**: Core functionality complete and tested
- 📈 **Roadmap**: Clear development phases planned
- 🐛 **Issues Welcome**: Report bugs and feature requests

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Based on Henry Bot M1** with significant architectural enhancements and RAG capabilities.
