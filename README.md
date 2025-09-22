# 🎨 CraftBuddy Backend
- Frontend Repo Link: Frontend(https://github.com/sajji18/app)
- Telegram Bot Repo Link: Bot(https://github.com/Pradipta-Sundar-Sahoo/Craftsbuddy.git)

**AI-Powered Artisan Marketplace Backend**

A sophisticated FastAPI-based backend system for connecting artisans with buyers through intelligent product discovery and personalized recommendations.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://postgresql.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-orange.svg)](https://openai.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-purple.svg)](https://pinecone.io)

## 🌟 Features

### 🤖 AI-Powered Buyer Query System
- **Natural Language Processing** with Google Gemini
- **Semantic Search** using OpenAI embeddings
- **Vector Similarity Search** with Pinecone
- **Multi-Agent Workflow** powered by LangGraph
- **Intent Detection & Sentiment Analysis**
- **Smart Product Recommendations**

### 🔐 Authentication & User Management
- **Multi-channel Authentication** (Email, Phone, Telegram)
- **OTP Verification** for secure access
- **JWT Token-based** session management
- **Role-based Access Control** (Buyers & Sellers)
- **Seller Onboarding** workflow

### 📦 Product Catalog Management
- **Product CRUD Operations** with image storage
- **Advanced Search & Filtering**
- **Category Management**
- **Inventory Tracking**
- **Cloud Image Storage** with Google Cloud

### 🏗️ Enterprise Architecture
- **RESTful API Design** with OpenAPI documentation
- **Database Migrations** with Alembic
- **Cloud-native** deployment ready
- **Comprehensive Logging** and monitoring
- **Scalable Infrastructure** design

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 15+
- Google Cloud Storage account
- OpenAI API key
- Pinecone account
- Google Gemini API key

### 1. Clone & Setup
```bash
git clone <repository-url>
cd CraftsBuddy-Backend
python -m venv myvenv
source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/craftbuddy

# Google Cloud Storage
GCS_BUCKET_NAME=your-gcs-bucket-name
GCS_CREDENTIALS_PATH=path/to/service-account.json

# AI Services
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX_NAME=craftbuddy-products
```

### 3. Database Setup
```bash
# Run migrations
python run_migration.py

# Or create tables directly
python create_tables.py
```

### 4. Initialize AI System
```bash
# Setup vector database and embeddings
python init_buyer_query_system.py
```

### 5. Start Server
```bash
# Development
python start_server.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

🎉 **Server running at**: http://localhost:8000
📚 **API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
CraftsBuddy-Backend/
├── 📂 src/
│   ├── 📂 apps/
│   │   ├── 🤖 buyer_query_agent.py    # AI-powered query system
│   │   ├── 🔐 login_service.py        # Authentication logic
│   │   ├── 📦 catalog_service.py      # Product management
│   │   ├── 👨‍💼 seller_service.py        # Seller operations
│   │   ├── 📱 otp_service.py          # OTP verification
│   │   ├── 🛣️ routes.py               # API endpoints
│   │   ├── 🗃️ db_models.py            # Database models
│   │   └── 📋 interface.py            # Pydantic models
│   ├── 📂 core/
│   │   ├── ⚙️ settings.py             # Configuration
│   │   ├── 🗄️ database.py            # Database connection
│   │   └── 🔧 database_service.py     # Database operations
│   └── 📂 utils/                      # Utility functions
├── 📂 alembic/                        # Database migrations
├── 🚀 main.py                         # FastAPI application
├── 🏃 start_server.py                 # Server startup script
├── 🤖 init_buyer_query_system.py      # AI system initialization
└── 📋 requirements.txt                # Python dependencies
```



## 🤖 AI Query System

The heart of CraftBuddy's intelligent search experience:

### Architecture
```
User Query → Gemini (Intent/Sentiment) → OpenAI (Embeddings) → Pinecone (Vector Search) → Results
```

### Workflow
1. **Preprocessing** - Clean and normalize queries
2. **Intent Detection** - Understand user intent (shopping, info, unclear)
3. **Sentiment Analysis** - Detect emotional context
4. **Embedding Generation** - Create semantic vectors with OpenAI
5. **Vector Search** - Find similar products with Pinecone
6. **Re-ranking** - Improve relevance with AI
7. **Related Products** - Suggest complementary items
8. **Response Generation** - Format intelligent responses

### Example Usage
```bash
POST /api/catalog/query
{
  "query": "I'm looking for handmade blue pottery for my kitchen"
}
```

Response:
```json
{
  "type": "products",
  "message": "Found 8 products matching your query",
  "products": [...],
  "related_products": [...],
  "query_metadata": {
    "intent": "shopping",
    "sentiment": "positive",
    "confidence": 0.92
  },
  "processing_time": 0.45
}
```

## 🗄️ Database Schema

### Core Tables
- **users** - User accounts and profiles
- **products** - Product catalog with specifications
- **user_sessions** - Authentication sessions
- **otp_verifications** - OTP tracking

### Key Features
- **UUID Primary Keys** for security
- **Soft Deletes** for data integrity
- **Audit Trails** with timestamps
- **JSON Fields** for flexible metadata
- **Indexes** for performance optimization

### Environment Variables
```bash
# Production settings
DATABASE_URL=postgresql://...
GEMINI_API_KEY=your-production-key
OPENAI_API_KEY=your-production-key
PINECONE_API_KEY=your-production-key
GCS_BUCKET_NAME=your-production-bucket
```

### Health Checks
- `GET /health` - System health status
- `GET /api/routes` - Available API routes

## 🔧 Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

## 📚 Documentation

- **API Docs**: Available at `/docs` (Swagger UI)
- **ReDoc**: Available at `/redoc`
- **Postman Collection**: `CraftBuddy_API_Postman_Collection.json`


**Built with ❤️ for artisans and craft lovers worldwide**


*Empowering creativity through intelligent technology* 🎨✨


