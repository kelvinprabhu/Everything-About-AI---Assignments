# 🍳 Multi-Agent Recipe Generation API

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An intelligent, multi-agent system for generating personalized recipes with nutritional analysis and automated shopping lists. Built with FastAPI, LangGraph, and Google's Gemini AI.

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### 🤖 Multi-Agent System
- **Planning Agent**: Creates recipe concepts and fetches nutritional data
- **Shopping Agent**: Generates smart shopping lists by comparing ingredients
- **Structuring Agent**: Formats everything into schema.org Recipe format

### 🎯 Core Capabilities
- ✅ **Smart Recipe Generation** based on available ingredients
- ✅ **Nutritional Analysis** with detailed macro/micronutrient breakdown
- ✅ **Automated Shopping Lists** categorized by store sections
- ✅ **Dietary Restriction Support** (Vegetarian, Vegan, Gluten-free, etc.)
- ✅ **Multiple Cuisines** (Indian, Italian, Chinese, Mexican, Thai, etc.)
- ✅ **Difficulty Levels** (Easy, Medium, Hard)
- ✅ **Flexible Servings** (1-12 people)
- ✅ **Time-based Planning** (10-240 minutes)

### 🚀 API Features
- RESTful API with OpenAPI/Swagger documentation
- CORS-enabled for frontend integration
- Request validation and error handling
- Health check endpoints
- UUID-based recipe tracking

---

## 🏗️ Architecture

```
┌─────────────────┐
│   FastAPI App   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│      LangGraph Workflow         │
├─────────────────────────────────┤
│  ┌──────────────────────────┐  │
│  │   Planning Agent         │  │
│  │   - Recipe concept       │  │
│  │   - Nutrition tool       │  │
│  └──────────┬───────────────┘  │
│             ▼                   │
│  ┌──────────────────────────┐  │
│  │   Shopping Agent         │  │
│  │   - Ingredient compare   │  │
│  │   - Shopping list gen    │  │
│  └──────────┬───────────────┘  │
│             ▼                   │
│  ┌──────────────────────────┐  │
│  │   Structuring Agent      │  │
│  │   - Schema.org format    │  │
│  │   - Final output         │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  External APIs  │
├─────────────────┤
│ - Google Gemini │
│ - CalorieNinja  │
└─────────────────┘
```

---

## 📦 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Windows, macOS, or Linux
- **Memory**: 2GB RAM minimum
- **Internet**: Required for API calls

### API Keys Required
1. **Google Gemini API Key** (for AI generation)
   - Get it at: https://makersuite.google.com/app/apikey
   
2. **CalorieNinja API Key** (for nutrition data)
   - Get it at: https://calorieninjas.com/api

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/kelvinprabhu/Everything-About-AI---Assignment.git
cd Google AI Studio Recipe Creator
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt:**
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
langchain==0.1.0
langchain-google-genai==0.0.6
langgraph==0.0.20
requests==2.31.0
python-multipart==0.0.6
```

### Step 4: Verify Installation

```bash
python -c "import fastapi, langchain, langgraph; print('✅ All packages installed!')"
```

---

## ⚙️ Configuration

### 1. Set Up API Keys

**Option A: Environment Variables (Recommended)**

**Windows:**
```bash
set GOOGLE_API_KEY=your_google_api_key_here
set NUTRITION_API_KEY=your_calorieninjas_key_here
```

**macOS/Linux:**
```bash
export GOOGLE_API_KEY=your_google_api_key_here
export NUTRITION_API_KEY=your_calorieninjas_key_here
```

**Option B: .env File**

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
NUTRITION_API_KEY=your_calorieninjas_key_here
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

Add to your main.py:
```python
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NUTRITION_API_KEY = os.getenv("NUTRITION_API_KEY")
```

### 2. Configure Application Settings

Edit `main.py` to customize:

```python
# CORS Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server Settings (at bottom of file)
uvicorn.run(
    "main:app",
    host="0.0.0.0",      # Listen on all interfaces
    port=8000,            # Port number
    reload=True,          # Auto-reload on code changes
    log_level="info"      # Logging level
)
```

---

## 🎮 Usage

### Starting the Server

**Development Mode (with auto-reload):**
```bash
python main.py
```

**Production Mode:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Custom Port:**
```bash
uvicorn main:app --port 8080
```

### Accessing the API

- **API Root**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 📖 API Documentation

### Endpoint Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/api/v1/recipe/generate` | Generate complete recipe |
| POST | `/api/v1/nutrition` | Get nutrition info |
| POST | `/api/v1/shopping-list` | Generate shopping list |
| GET | `/api/v1/cuisines` | List available cuisines |
| GET | `/api/v1/dietary-restrictions` | List dietary options |
| GET | `/api/v1/difficulty-levels` | List difficulty levels |

### Example Requests

#### 1. Generate Recipe

**cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/recipe/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "available_ingredients": "rice, tomatoes, onions, garlic, eggs, butter, milk",
    "dietary_restrictions": ["Vegetarian"],
    "cuisine": "South Indian",
    "difficulty": "Medium",
    "servings": 4,
    "cooking_time_minutes": 45
  }'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/recipe/generate",
    json={
        "available_ingredients": "rice, tomatoes, onions, garlic, eggs",
        "dietary_restrictions": ["Vegetarian"],
        "cuisine": "Italian",
        "difficulty": "Easy",
        "servings": 2,
        "cooking_time_minutes": 30
    }
)

recipe = response.json()
print(f"Recipe: {recipe['recipe']['name']}")
```

**JavaScript:**
```javascript
fetch('http://localhost:8000/api/v1/recipe/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    available_ingredients: 'pasta, tomatoes, garlic, olive oil',
    dietary_restrictions: ['Vegan'],
    cuisine: 'Italian',
    difficulty: 'Easy',
    servings: 2,
    cooking_time_minutes: 20
  })
})
.then(response => response.json())
.then(data => console.log('Recipe:', data.recipe.name));
```

#### 2. Get Nutrition Info

```bash
curl -X POST "http://localhost:8000/api/v1/nutrition" \
  -H "Content-Type: application/json" \
  -d '{
    "ingredients": "2 cups rice, 1 lb chicken, 3 tomatoes, 2 onions"
  }'
```

#### 3. Generate Shopping List

```bash
curl -X POST "http://localhost:8000/api/v1/shopping-list" \
  -H "Content-Type: application/json" \
  -d '{
    "recipe_ingredients": [
      "2 cups rice",
      "500g chicken",
      "3 tomatoes",
      "2 onions",
      "4 cloves garlic"
    ],
    "available_ingredients": "rice, onions, salt, pepper"
  }'
```

### Response Format

**Success Response:**
```json
{
  "recipe_id": "550e8400-e29b-41d4-a716-446655440000",
  "recipe": {
    "@context": "https://schema.org",
    "@type": "Recipe",
    "name": "Spicy Tomato Rice",
    "description": "A flavorful South Indian rice dish",
    "cuisine": "South Indian",
    "difficulty": "Medium",
    "prepTime": "PT15M",
    "cookTime": "PT30M",
    "totalTime": "PT45M",
    "recipeYield": "4",
    "recipeIngredient": [...],
    "recipeInstructions": [...],
    "nutrition": {
      "calories": 350.5,
      "protein_g": 12.3,
      "fat_total_g": 8.5,
      "carbohydrates_total_g": 58.2
    },
    "shoppingList": [...],
    "availableItems": [...]
  },
  "generation_time": "2024-11-18T10:30:45.123456",
  "status": "success"
}
```

**Error Response:**
```json
{
  "error": "Recipe generation failed",
  "detail": "Invalid ingredients format",
  "timestamp": "2024-11-18T10:30:45.123456"
}
```

---

## 📁 Project Structure

```
Everything-About-AI---Assignments/Google AI Studio Recipe Creator/
│
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── MultiAgentRecipeGenerationSystemLangGraph.py # function file with recipe generator base code 
│
├── README.md                 # This file
│
├── test/                    # Test directory
│   ├── mark1.ipynb
│   ├── mark2.ipynb
│   └── rg2.py
│
├── screenshot/
|    ├── Capture 001 -Multi-Agent Recipe Generation API.pdf #docs page shreenshot 
├── output/
|   ├── test outputs 

```

---

## 🔧 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Code Formatting

```bash
# Install formatters
pip install black isort flake8

# Format code
black main.py
isort main.py

# Check style
flake8 main.py
```

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

### Hot Reload

Development server automatically reloads on code changes:

```bash
uvicorn main:app --reload
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
pip install -r requirements.txt
```

#### 2. API Key Errors

**Problem:** `Authentication failed` or `Invalid API key`

**Solution:**
- Verify your API keys are correct
- Check environment variables are set
- Ensure no extra spaces in keys

```bash
# Check environment variables
echo $GOOGLE_API_KEY
echo $NUTRITION_API_KEY
```

#### 3. Port Already in Use

**Problem:** `Address already in use`

**Solution:**
```bash
# Kill process on port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:8000 | xargs kill -9
```

#### 4. CORS Errors

**Problem:** `CORS policy: No 'Access-Control-Allow-Origin' header`

**Solution:**
Update CORS settings in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify your domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 5. Slow Response Times

**Problem:** API takes too long to respond

**Solution:**
- Increase timeout settings
- Use async operations
- Implement caching
- Reduce number of agents/tool calls

#### 6. Nutrition API Rate Limits

**Problem:** `Rate limit exceeded`

**Solution:**
- Implement rate limiting on your end
- Cache nutrition data
- Upgrade to paid CalorieNinja plan

---

## 🚦 Best Practices

### 1. Security
- **Never commit API keys** to version control
- Use environment variables
- Implement rate limiting
- Add authentication for production

### 2. Performance
- Use connection pooling
- Implement caching (Redis)
- Use async/await properly
- Monitor resource usage

### 3. Error Handling
- Log all errors
- Provide meaningful error messages
- Implement retry logic
- Use circuit breakers

### 4. Testing
- Write unit tests for all endpoints
- Test with different inputs
- Test error scenarios
- Load test for production

---

## 🔒 Security Considerations

### Production Checklist

- [ ] Remove hardcoded API keys
- [ ] Add authentication (JWT, OAuth)
- [ ] Implement rate limiting
- [ ] Enable HTTPS only
- [ ] Add input sanitization
- [ ] Set up monitoring
- [ ] Configure logging
- [ ] Add request validation
- [ ] Set up error tracking (Sentry)
- [ ] Implement CSRF protection

### Example: Adding Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

@app.post("/api/v1/recipe/generate", dependencies=[Depends(verify_token)])
async def generate_recipe_endpoint(request: RecipeRequest):
    # Your code here
    pass
```

---

## 📊 Monitoring

### Health Checks

```bash
# Simple health check
curl http://localhost:8000/health

# Detailed monitoring
curl http://localhost:8000/health | jq
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics (Optional)

```bash
pip install prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide
- Write tests for new features
- Update documentation
- Add comments for complex logic

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FastAPI** - Modern web framework
- **LangChain** - LLM orchestration
- **LangGraph** - Multi-agent workflows
- **Google Gemini** - AI generation
- **CalorieNinja** - Nutrition data

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/kelvinprabhu/Everything-About-AI---Assignments)
- **Email**: kelvinprabhu2071@gmail.com

---

## 🗺️ Roadmap

- [ ] Add recipe image generation
- [ ] Implement user authentication
- [ ] Add recipe rating system
- [ ] Support for more cuisines
- [ ] Mobile app integration
- [ ] Recipe sharing features
- [ ] Meal planning calendar
- [ ] Grocery delivery integration

---

## 📈 Version History

- **v1.0.0** (2024-11-18)
  - Initial release
  - Multi-agent recipe generation
  - Nutrition analysis
  - Shopping list generation
  - RESTful API with OpenAPI docs

---

**Made by Kelvin**