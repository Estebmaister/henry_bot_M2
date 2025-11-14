# Henry Bot M2 Development Workflow Guide

## 🎯 Overview

This guide outlines the complete development workflow for Henry Bot M2, a production-ready LLM agent with clean architecture principles. The workflow ensures code quality, comprehensive testing, and smooth deployment processes.

## 🏗️ Architecture Overview

### Clean Architecture Implementation
```
┌─────────────────────────────────────────┐
│           API Layer (FastAPI)           │  ← External interfaces
├─────────────────────────────────────────┤
│         Application Layer               │  ← Business logic
│  ┌─────────────┐  ┌─────────────────┐   │
│  │   Agent     │  │   Prompting     │   │
│  │ Orchestrator│  │    Engine       │   │
│  └─────────────┘  └─────────────────┘   │
├─────────────────────────────────────────┤
│         Infrastructure Layer            │  ← External services
│  ┌─────────────┐  ┌─────────────────┐   │
│  │     LLM     │  │      RAG        │   │
│  │ Integration │  │   System (P2)   │   │
│  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────┘
```

### Module Responsibilities
- **`src/core/`**: Core business logic and orchestration
- **`src/modules/api/`**: FastAPI server and HTTP interfaces
- **`src/modules/prompting/`**: Prompt engineering techniques
- **`src/modules/rag/`**: Document retrieval (Phase 2)
- **`src/modules/metrics/`**: Performance tracking
- **`src/modules/logging/`**: Structured logging

## 🚀 Development Setup

### Prerequisites
- Python 3.9+
- OpenRouter API key
- Git
- Docker (optional)

### Quick Setup
```bash
# 1. Clone and setup
git clone https://github.com/estebmaister/henry_bot_M2.git
cd henry_bot_M2
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start development server
python -m src.main server

# 5. Run tests
./quick_test.sh
```

## 🔄 Development Workflow

### Phase 1: Feature Development

#### 1.1 Create Feature Branch
```bash
git checkout -b feature/new-feature-name
```

#### 1.2 Development Guidelines
```bash
# Make changes following these principles:
# - Update src/core/ for business logic
# - Update src/modules/ for component changes
# - Add type hints to all functions
# - Include comprehensive docstrings
# - Handle errors appropriately
# - Add logging where needed
```

#### 1.3 Code Quality Standards
```python
# Example function with proper typing and documentation
def process_question(
    self,
    user_question: str,
    prompt_technique: Optional[str] = None,
    use_rag: bool = True
) -> Dict:
    """
    Process a user question with optional RAG augmentation.

    Args:
        user_question: The user's question
        prompt_technique: Prompting technique to use
        use_rag: Whether to use RAG system for context

    Returns:
        Dictionary containing the answer, metrics, and RAG scores

    Raises:
        LLMError: When LLM API call fails
        RAGError: When RAG retrieval fails
    """
```

### Phase 2: Testing

#### 2.1 Run Local Tests
```bash
# Quick test of all endpoints
./quick_test.sh

# Comprehensive test suite
python3 test_api.py

# Test specific functionality
python3 test_api.py --test chat
```

#### 2.2 Manual Testing Checklist
- [ ] Health endpoint returns 200
- [ ] Chat endpoint works with all prompt techniques
- [ ] Authentication works correctly
- [ ] Error handling returns proper responses
- [ ] Documentation loads correctly
- [ ] Metrics are logged properly

#### 2.3 Performance Testing
```bash
# Test response times
time curl -X POST "http://0.0.0.0:8000/api/v1/chat" \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"question": "Test question", "prompt_technique": "simple"}'
```

### Phase 3: Code Review

#### 3.1 Self-Review Checklist
- [ ] Code follows project architecture patterns
- [ ] Type hints are complete and accurate
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate
- [ ] Documentation is clear
- [ ] Tests cover new functionality

#### 3.2 Architecture Review
```python
# Questions to ask during review:
# - Does this change respect clean architecture?
# - Are dependencies properly injected?
# - Is the module cohesive?
# - Are interfaces clear and minimal?
# - Does this follow SOLID principles?
```

### Phase 4: Integration & Deployment

#### 4.1 Update Documentation
```bash
# Update relevant documentation:
# - README.md for user-facing changes
# - API_TESTING_GUIDE.md for new endpoints
# - CLAUDE.md for architectural changes
```

#### 4.2 Pre-deployment Testing
```bash
# Full test suite
python3 test_api.py

# Docker testing
docker build -t henry-bot-m2-test .
docker run -p 8001:8000 \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -e API_KEY="$API_KEY" \
  henry-bot-m2-test

# Test Docker instance
curl -X GET "http://localhost:8001/health"
```

#### 4.3 Deployment
```bash
# Merge to main branch
git checkout main
git merge feature/new-feature-name

# Tag release
git tag -a v2.x.x -m "Release version 2.x.x"

# Push to production
git push origin main --tags
```

## 🧪 Testing Strategy

### Test Pyramid
```
        ┌─────────────────┐
        │  E2E Tests      │  ← Few, comprehensive API tests
        ├─────────────────┤
        │ Integration     │  ← Component interaction tests
        ├─────────────────┤
        │   Unit Tests    │  ← Many, focused tests (planned)
        └─────────────────┘
```

### Current Testing Tools
1. **`quick_test.sh`** - Fast endpoint verification
2. **`test_api.py`** - Comprehensive API testing
3. **`henry_bot_postman_collection.json`** - Visual testing
4. **Manual testing** - Interactive documentation

### Adding New Tests
```python
# When adding new endpoints, update:
# 1. test_api.py - Add new test methods
# 2. quick_test.sh - Add curl commands
# 3. Postman collection - Add new requests
# 4. API_TESTING_GUIDE.md - Document new tests
```

## 📝 Development Patterns

### 1. Configuration Management
```python
# Use settings for all configuration
from src.core.config import settings

class NewService:
    def __init__(self):
        self.timeout = settings.request_timeout
        self.api_key = settings.openrouter_api_key
```

### 2. Error Handling
```python
# Use custom exceptions with proper logging
from src.core.exceptions import HenryBotError
from src.modules.logging.logger import log_error

try:
    result = risky_operation()
except SpecificError as e:
    log_error(
        error_type="SpecificError",
        error_message=str(e),
        context={"additional": "context"}
    )
    raise HenryBotError(f"Operation failed: {e}")
```

### 3. Metrics Tracking
```python
# Track performance for all operations
from src.modules.metrics.tracker import track_api_call

def process_request():
    tracker = track_api_call(operation="my_operation")
    try:
        result = do_work()
        tracker.stop()
        return result
    except Exception as e:
        tracker.stop()
        raise
```

### 4. API Endpoint Pattern
```python
# Follow established patterns for new endpoints
@app.post("/api/v1/new-endpoint", response_model=ResponseModel)
async def new_endpoint(request: RequestModel):
    """Documentation for the endpoint."""
    try:
        # Validate authentication
        # Process request
        # Return response
        return ResponseModel(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 🔧 Common Development Tasks

### Adding a New Prompt Technique
```python
# 1. Update src/modules/prompting/engine.py
def build_new_technique_prompt(
    self,
    user_question: str,
    rag_context: Optional[str] = None
) -> List[Dict[str, str]]:
    # Implement new technique

# 2. Update create_prompt function
def create_prompt(
    user_question: str,
    technique: str = "few_shot",  # Add new technique option
    rag_context: Optional[str] = None
) -> List[Dict[str, str]]:
    if technique == "new_technique":
        return builder.build_new_technique_prompt(user_question, rag_context)

# 3. Update API schemas
# Add "new_technique" to enum in schemas.py
```

### Adding a New API Endpoint
```python
# 1. Define request/response models
class NewRequest(BaseModel):
    field1: str
    field2: Optional[int] = None

class NewResponse(BaseModel):
    result: str
    metrics: Dict[str, Any]

# 2. Add endpoint to server.py
@app.post("/api/v1/new-endpoint", response_model=NewResponse)
async def new_endpoint(request: NewRequest):
    # Implementation

# 3. Update tests
# Add new endpoint to test_api.py
# Add curl command to quick_test.sh
# Add to Postman collection
```

### Adding New Metrics
```python
# 1. Extend tracker functionality
class MetricsTracker:
    def __init__(self):
        self.custom_metric = 0

    def track_custom_event(self, value: int):
        self.custom_metric += value

# 2. Update CSV logging
# Add new columns to metrics export format
```

## 🚨 Troubleshooting Workflow

### 1. Identify the Issue
```bash
# Check server status
curl -X GET "http://0.0.0.0:8000/health"

# Check logs
tail -f logs/app.log

# Check metrics
tail -f logs/metrics.csv
```

### 2. Isolate the Problem
```bash
# Test individual components
python3 test_api.py --test health
python3 test_api.py --test auth
python3 test_api.py --test chat
```

### 3. Debug Step by Step
```bash
# Test with minimal configuration
export MODEL_NAME="simple-model"
python -m src.main server

# Test with different API keys
export API_KEY="test-key"
./quick_test.sh
```

### 4. Common Fixes
```bash
# Restart server after config changes
pkill -f "python -m src.main"
python -m src.main server

# Clear cached data
rm -rf data/__pycache__/
rm -rf logs/metrics.csv

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📊 Monitoring & Maintenance

### Daily Checks
```bash
# Health check
curl -X GET "http://0.0.0.0:8000/health"

# Log monitoring
tail -n 100 logs/app.log | grep ERROR

# Performance metrics
tail -n 50 logs/metrics.csv
```

### Weekly Maintenance
```bash
# Log rotation
mv logs/app.log logs/app.log.$(date +%Y%m%d)
touch logs/app.log

# Dependency updates
pip list --outdated
pip install --upgrade package-name
```

### Performance Monitoring
```bash
# Monitor response times
./quick_test.sh | grep "Response time"

# Check error rates
grep "ERROR" logs/app.log | wc -l

# Track API usage
tail -f logs/metrics.csv | awk -F, '{print $1","$2}'
```

## 🎯 Development Best Practices

### 1. Code Organization
- Keep modules focused on single responsibilities
- Use dependency injection for testability
- Follow established naming conventions
- Maintain clear separation between layers

### 2. Testing Strategy
- Test all public API endpoints
- Include both success and failure cases
- Measure performance characteristics
- Test with different configurations

### 3. Documentation
- Keep README.md up to date
- Document all API changes
- Include examples in docstrings
- Update testing guides

### 4. Security
- Never commit API keys
- Use environment variables for secrets
- Validate all user inputs
- Follow security best practices

### 5. Performance
- Monitor response times
- Track token usage and costs
- Optimize database queries
- Use caching where appropriate

---

## 🚀 Quick Reference

### Development Commands
```bash
# Start development server
python -m src.main server

# Run tests
./quick_test.sh
python3 test_api.py

# Check status
python -m src.main status

# CLI testing
python -m src.main cli "test question"

# View logs
tail -f logs/app.log
tail -f logs/metrics.csv
```

### Testing URLs
- **API Documentation**: http://0.0.0.0:8000/docs
- **Health Check**: http://0.0.0.0:8000/health
- **ReDoc**: http://0.0.0.0:8000/redoc

### Key Files
- **`src/core/agent.py`** - Main business logic
- **`src/modules/api/server.py`** - API endpoints
- **`src/core/config.py`** - Configuration management
- **`.env`** - Environment variables
- **`logs/app.log`** - Application logs
- **`logs/metrics.csv`** - Performance metrics

---

`★ Insight ─────────────────────────────────────`
- **Clean Architecture**: Maintain separation between layers and modules
- **Testing First**: Always write tests before deploying changes
- **Monitoring**: Track performance and errors continuously
- **Documentation**: Keep docs in sync with code changes
`─────────────────────────────────────────────────`