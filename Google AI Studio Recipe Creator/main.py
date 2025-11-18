"""
FastAPI Application for Multi-Agent Recipe Generation System
Provides REST API endpoints for recipe generation, nutrition info, and shopping lists
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from enum import Enum
import uvicorn
from datetime import datetime
import uuid
import json

# Import the recipe generation system
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import requests
import operator

# =============================================================================
# CONFIGURATION
# =============================================================================
GOOGLE_API_KEY = "AIzaSyBq3XbBUGmQDZefsnpkA2EyUs_4t52PacE"
NUTRITION_API_KEY = "O6D2W2XklbtxEUvB2uDiBg==7aDTZgDoxbQZh60u"

# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(
    title="Multi-Agent Recipe Generation API",
    description="AI-powered recipe generation with nutrition analysis and shopping lists",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ENUMS
# =============================================================================

class CuisineType(str, Enum):
    SOUTH_INDIAN = "South Indian"
    NORTH_INDIAN = "North Indian"
    ITALIAN = "Italian"
    CHINESE = "Chinese"
    MEXICAN = "Mexican"
    THAI = "Thai"
    JAPANESE = "Japanese"
    MEDITERRANEAN = "Mediterranean"
    AMERICAN = "American"
    FRENCH = "French"

class DifficultyLevel(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

class DietaryRestriction(str, Enum):
    VEGETARIAN = "Vegetarian"
    VEGAN = "Vegan"
    GLUTEN_FREE = "Gluten-free"
    DAIRY_FREE = "Dairy-free"
    NUT_FREE = "Nut-free"
    DIABETIC = "Diabetic"
    HALAL = "Halal"
    KOSHER = "Kosher"
    LOW_CALORIE = "Low-calorie"
    LOW_FAT = "Low-fat"
    LOW_SODIUM = "Low-sodium"

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class RecipeRequest(BaseModel):
    """Request model for recipe generation"""
    available_ingredients: str = Field(
        ...,
        description="Comma-separated list of available ingredients",
        example="rice, tomatoes, onions, garlic, eggs, butter"
    )
    dietary_restrictions: List[DietaryRestriction] = Field(
        default=[],
        description="List of dietary restrictions"
    )
    cuisine: CuisineType = Field(
        default=CuisineType.SOUTH_INDIAN,
        description="Type of cuisine"
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Recipe difficulty level"
    )
    servings: int = Field(
        default=2,
        ge=1,
        le=12,
        description="Number of servings (1-12)"
    )
    cooking_time_minutes: int = Field(
        default=50,
        ge=10,
        le=240,
        description="Target cooking time in minutes (10-240)"
    )

    @validator('available_ingredients')
    def validate_ingredients(cls, v):
        if not v or not v.strip():
            raise ValueError("Available ingredients cannot be empty")
        return v.strip()


class NutritionRequest(BaseModel):
    """Request model for nutrition information"""
    ingredients: str = Field(
        ...,
        description="Comma-separated list of ingredients with quantities",
        example="2 cups rice, 1 lb chicken, 3 tomatoes"
    )


class ShoppingListRequest(BaseModel):
    """Request model for shopping list generation"""
    recipe_ingredients: List[str] = Field(
        ...,
        description="List of ingredients needed for recipe"
    )
    available_ingredients: str = Field(
        ...,
        description="Comma-separated list of available ingredients"
    )


class RecipeResponse(BaseModel):
    """Response model for recipe generation"""
    recipe_id: str
    recipe: Dict
    generation_time: str
    status: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: str


# =============================================================================
# TOOLS (from original system)
# =============================================================================

@tool(description="Get detailed nutrition information for recipe ingredients")
def get_nutrition_info(ingredients_text: str) -> dict:
    """Fetches nutritional data for given ingredients"""
    api_url = 'https://api.calorieninjas.com/v1/nutrition?query='
    url = api_url + requests.utils.quote(ingredients_text)
    
    try:
        response = requests.get(url, headers={'X-Api-Key': NUTRITION_API_KEY})
        
        if response.status_code != 200:
            return {
                "error": True,
                "status_code": response.status_code,
                "message": "Failed to fetch nutrition data"
            }
        
        data = response.json()
        items = data.get("items", [])
        
        total_nutrition = {
            "calories": sum(item.get("calories", 0) for item in items),
            "protein_g": sum(item.get("protein_g", 0) for item in items),
            "fat_total_g": sum(item.get("fat_total_g", 0) for item in items),
            "carbohydrates_total_g": sum(item.get("carbohydrates_total_g", 0) for item in items),
            "fiber_g": sum(item.get("fiber_g", 0) for item in items),
            "sugar_g": sum(item.get("sugar_g", 0) for item in items),
            "sodium_mg": sum(item.get("sodium_mg", 0) for item in items)
        }
        
        return {
            "success": True,
            "nutrition": total_nutrition,
            "items_count": len(items)
        }
    except Exception as e:
        return {
            "error": True,
            "message": str(e)
        }


@tool(description="Compare recipe ingredients with available ingredients")
def compare_and_generate_shopping_list(
    recipe_ingredients: List[str], 
    available_ingredients: str
) -> dict:
    """Compares recipe ingredients with available ingredients"""
    
    def extract_ingredient_name(ingredient_text: str) -> str:
        words = ingredient_text.lower().split()
        quantity_words = {'cup', 'cups', 'tablespoon', 'tablespoons', 'tbsp', 'tsp', 
                         'teaspoon', 'teaspoons', 'oz', 'lb', 'lbs', 'gram', 'grams', 
                         'g', 'kg', 'ml', 'l', 'liter', 'liters', 'piece', 'pieces',
                         'pcs', 'of', 'medium', 'large', 'small', 'fresh', 'dried'}
        
        ingredient_words = [w for w in words if not w.replace('.', '').isdigit() 
                          and w not in quantity_words]
        
        return ' '.join(ingredient_words) if ingredient_words else ingredient_text.lower()
    
    def categorize_ingredient(ingredient: str) -> str:
        ingredient_lower = ingredient.lower()
        categories = {
            "Produce": ["tomato", "onion", "garlic", "potato", "vegetable", "fruit"],
            "Grains": ["rice", "flour", "pasta", "wheat", "bread"],
            "Dairy": ["milk", "butter", "cheese", "yogurt", "cream"],
            "Proteins": ["egg", "chicken", "meat", "fish", "tofu", "beans"],
            "Spices": ["pepper", "salt", "turmeric", "cumin", "spice", "herb"],
            "Oils & Condiments": ["oil", "vinegar", "sauce", "condiment"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in ingredient_lower for keyword in keywords):
                return category
        return "Other"
    
    available_list = [item.strip().lower() for item in available_ingredients.split(',')]
    
    shopping_list = []
    already_available = []
    
    for ingredient in recipe_ingredients:
        category = categorize_ingredient(ingredient)
        ingredient_name = extract_ingredient_name(ingredient)
        
        is_available = any(
            ingredient_name in avail or avail in ingredient_name 
            for avail in available_list
        )
        
        if is_available:
            already_available.append({
                "item": ingredient,
                "category": category,
                "status": "available"
            })
        else:
            shopping_list.append({
                "item": ingredient,
                "category": category,
                "status": "need_to_purchase"
            })
    
    shopping_by_category = {}
    for item in shopping_list:
        category = item['category']
        if category not in shopping_by_category:
            shopping_by_category[category] = []
        shopping_by_category[category].append(item['item'])
    
    return {
        "items_to_purchase": shopping_list,
        "items_available": already_available,
        "shopping_by_category": shopping_by_category,
        "total_items_needed": len(recipe_ingredients),
        "items_to_buy": len(shopping_list),
        "items_in_stock": len(already_available)
    }


# =============================================================================
# RECIPE GENERATION LOGIC (Simplified from original)
# =============================================================================

class RecipeState(TypedDict):
    """State for recipe generation"""
    available_ingredients: str
    dietary_restrictions: List[str]
    cuisine: str
    difficulty: str
    servings: int
    cooking_time_minutes: int
    recipe_plan: Optional[str]
    nutrition_data: Optional[Dict]
    shopping_data: Optional[Dict]
    final_recipe: Optional[Dict]
    messages: Annotated[List, operator.add]
    current_agent: str
    errors: Annotated[List[str], operator.add]



def get_llm(temperature: float = 0.7, tools: Optional[List] = None):
    """Get configured LLM instance"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature
    )
    if tools:
        llm = llm.bind_tools(tools)
    return llm


def planning_agent(state: RecipeState) -> RecipeState:
    """Planning & Nutrition Agent"""
    llm = get_llm(temperature=0.7, tools=[get_nutrition_info])
    
    system_message = """You are a professional chef and recipe planner.
Create a recipe concept, list ingredients with quantities, and get nutrition info."""
    
    user_message = f"""Create a recipe plan:
Available: {state['available_ingredients']}
Dietary: {', '.join(state['dietary_restrictions'])}
Cuisine: {state['cuisine']}
Difficulty: {state['difficulty']}
Servings: {state['servings']}
Time: {state['cooking_time_minutes']} minutes

List ingredients with quantities and use get_nutrition_info tool."""
    
    messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]
    response = llm.invoke(messages)
    
    nutrition_data = None
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'get_nutrition_info':
                result = get_nutrition_info.invoke(tool_call['args'])
                nutrition_data = result
    
    state['recipe_plan'] = response.content
    state['nutrition_data'] = nutrition_data
    state['messages'].append(AIMessage(content=response.content))
    state['current_agent'] = 'planning'
    return state


def shopping_agent(state: RecipeState) -> RecipeState:
    """Shopping & Requirements Agent"""
    llm = get_llm(temperature=0.5, tools=[compare_and_generate_shopping_list])
    
    system_message = """Extract ingredients and use compare_and_generate_shopping_list tool."""
    
    user_message = f"""Recipe: {state['recipe_plan']}
Available: {state['available_ingredients']}

Extract ingredients and call compare_and_generate_shopping_list."""
    
    messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]
    response = llm.invoke(messages)
    
    shopping_data = None
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'compare_and_generate_shopping_list':
                result = compare_and_generate_shopping_list.invoke(tool_call['args'])
                shopping_data = result
    
    if not shopping_data:
        shopping_data = {
            "items_to_purchase": [],
            "items_available": [],
            "shopping_by_category": {},
            "total_items_needed": 0,
            "items_to_buy": 0,
            "items_in_stock": 0
        }
    
    state['shopping_data'] = shopping_data
    state['messages'].append(AIMessage(content=response.content))
    state['current_agent'] = 'shopping'
    return state


def structuring_agent(state: RecipeState) -> RecipeState:
    """Structuring Agent"""
    llm = get_llm(temperature=0.1)
    
    nutrition_data = state.get('nutrition_data', {})
    if nutrition_data and nutrition_data.get('success'):
        per_serving = {k: round(v / state['servings'], 2) 
                      for k, v in nutrition_data['nutrition'].items()}
    else:
        per_serving = {
            "calories": 350.0, "protein_g": 15.0, "fat_total_g": 12.0,
            "carbohydrates_total_g": 45.0, "fiber_g": 5.0,
            "sugar_g": 3.0, "sodium_mg": 400.0
        }
    
    shopping_data = state.get('shopping_data', {})
    
    recipe_dict = {
        "@context": "https://schema.org",
        "@type": "Recipe",
        "name": f"{state['cuisine']} Recipe",
        "description": "AI-generated recipe",
        "author": "AI Chef",
        "cuisine": state['cuisine'],
        "difficulty": state['difficulty'],
        "prepTime": f"PT{state['cooking_time_minutes']//2}M",
        "cookTime": f"PT{state['cooking_time_minutes']//2}M",
        "totalTime": f"PT{state['cooking_time_minutes']}M",
        "recipeYield": str(state['servings']),
        "recipeIngredient": [],
        "recipeInstructions": [],
        "nutrition": per_serving,
        "shoppingList": shopping_data.get('items_to_purchase', []),
        "availableItems": shopping_data.get('items_available', []),
        "shoppingByCategory": shopping_data.get('shopping_by_category', {}),
        "keywords": [state['cuisine'], state['difficulty']],
        "datePublished": datetime.now().strftime("%Y-%m-%d"),
        "image": "https://via.placeholder.com/800x600"
    }
    
    state['final_recipe'] = recipe_dict
    state['current_agent'] = 'structuring'
    return state


def build_recipe_graph():
    """Build LangGraph workflow"""
    workflow = StateGraph(RecipeState)
    workflow.add_node("planning", planning_agent)
    workflow.add_node("shopping", shopping_agent)
    workflow.add_node("structuring", structuring_agent)
    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "shopping")
    workflow.add_edge("shopping", "structuring")
    workflow.add_edge("structuring", END)
    return workflow.compile()


def generate_recipe_internal(
    available_ingredients: str,
    dietary_restrictions: List[str],
    cuisine: str,
    difficulty: str,
    servings: int,
    cooking_time_minutes: int
) -> Dict:
    """Internal recipe generation"""
    initial_state: RecipeState = {
        "available_ingredients": available_ingredients,
        "dietary_restrictions": dietary_restrictions,
        "cuisine": cuisine,
        "difficulty": difficulty,
        "servings": servings,
        "cooking_time_minutes": cooking_time_minutes,
        "recipe_plan": None,
        "nutrition_data": None,
        "shopping_data": None,
        "final_recipe": None,
        "messages": [],
        "current_agent": "start",
        "errors": []
    }
    
    graph = build_recipe_graph()
    final_state = graph.invoke(initial_state)
    return final_state['final_recipe']


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Agent Recipe Generation API",
        "version": "1.0.0",
        "endpoints": {
            "generate_recipe": "/api/v1/recipe/generate",
            "nutrition": "/api/v1/nutrition",
            "shopping_list": "/api/v1/shopping-list",
            "health": "/health"
        },
        "documentation": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "recipe-generation-api"
    }


@app.post(
    "/api/v1/recipe/generate",
    response_model=RecipeResponse,
    tags=["Recipe"],
    summary="Generate a complete recipe"
)
async def generate_recipe_endpoint(request: RecipeRequest):
    """
    Generate a complete recipe with nutrition info and shopping list
    
    - **available_ingredients**: Comma-separated ingredients you have
    - **dietary_restrictions**: List of dietary restrictions
    - **cuisine**: Type of cuisine
    - **difficulty**: Recipe difficulty level
    - **servings**: Number of servings
    - **cooking_time_minutes**: Target cooking time
    """
    try:
        recipe_id = str(uuid.uuid4())
        
        recipe = generate_recipe_internal(
            available_ingredients=request.available_ingredients,
            dietary_restrictions=[dr.value for dr in request.dietary_restrictions],
            cuisine=request.cuisine.value,
            difficulty=request.difficulty.value,
            servings=request.servings,
            cooking_time_minutes=request.cooking_time_minutes
        )
        
        return RecipeResponse(
            recipe_id=recipe_id,
            recipe=recipe,
            generation_time=datetime.now().isoformat(),
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recipe generation failed: {str(e)}"
        )


@app.post(
    "/api/v1/nutrition",
    tags=["Nutrition"],
    summary="Get nutrition information"
)
async def get_nutrition_endpoint(request: NutritionRequest):
    """
    Get nutritional information for given ingredients
    
    - **ingredients**: Comma-separated list with quantities
    """
    try:
        result = get_nutrition_info.invoke({"ingredients_text": request.ingredients})
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Nutrition lookup failed: {str(e)}"
        )


@app.post(
    "/api/v1/shopping-list",
    tags=["Shopping"],
    summary="Generate shopping list"
)
async def generate_shopping_list_endpoint(request: ShoppingListRequest):
    """
    Generate shopping list by comparing recipe vs available ingredients
    
    - **recipe_ingredients**: List of ingredients needed
    - **available_ingredients**: Comma-separated available ingredients
    """
    try:
        result = compare_and_generate_shopping_list.invoke({
            "recipe_ingredients": request.recipe_ingredients,
            "available_ingredients": request.available_ingredients
        })
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Shopping list generation failed: {str(e)}"
        )


@app.get("/api/v1/cuisines", tags=["Reference"])
async def get_cuisines():
    """Get available cuisine types"""
    return {"cuisines": [c.value for c in CuisineType]}


@app.get("/api/v1/dietary-restrictions", tags=["Reference"])
async def get_dietary_restrictions():
    """Get available dietary restrictions"""
    return {"dietary_restrictions": [d.value for d in DietaryRestriction]}


@app.get("/api/v1/difficulty-levels", tags=["Reference"])
async def get_difficulty_levels():
    """Get available difficulty levels"""
    return {"difficulty_levels": [d.value for d in DifficultyLevel]}


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )