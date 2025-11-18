"""
Multi-Agent Recipe Generation System using LangGraph
Agents:
1. Planning Agent - Creates recipe plan and gets nutrition info
2. Shopping & Requirements Agent - Generates shopping lists and additional requirements
3. Structuring Agent - Formats everything into final recipe schema
"""

from typing import TypedDict, List, Dict, Optional, Annotated, Union
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator
import requests
import json
import operator

# =============================================================================
# CONFIGURATION
# =============================================================================
GOOGLE_API_KEY = ""
NUTRITION_API_KEY = ""

# =============================================================================
# TOOLS DEFINITION
# =============================================================================

@tool(description="Get detailed nutrition information for recipe ingredients")
def get_nutrition_info(ingredients_text: str) -> dict:
    """
    Fetches nutritional data for given ingredients.
    Args:
        ingredients_text: Comma-separated list of ingredients with quantities
    Returns:
        Dictionary with nutrition data including calories, protein, fat, carbs
    """
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
        
        # Aggregate nutrition data
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


@tool(description="Compare recipe ingredients with available ingredients and generate shopping list")
def compare_and_generate_shopping_list(
    recipe_ingredients: List[str], 
    available_ingredients: str
) -> dict:
    """
    Compares recipe ingredients with available ingredients and generates shopping list.
    Args:
        recipe_ingredients: List of ingredients needed for recipe with quantities
        available_ingredients: Comma-separated string of available ingredients
    Returns:
        Dictionary with items to purchase and items already available
    """
    # Parse available ingredients into a searchable format
    available_list = [item.strip().lower() for item in available_ingredients.split(',')]
    
    # Create a more flexible matching system
    def extract_ingredient_name(ingredient_text: str) -> str:
        """Extract the main ingredient name from text like '2 cups rice'"""
        # Remove common quantity words and numbers
        words = ingredient_text.lower().split()
        quantity_words = {'cup', 'cups', 'tablespoon', 'tablespoons', 'tbsp', 'tsp', 
                         'teaspoon', 'teaspoons', 'oz', 'lb', 'lbs', 'gram', 'grams', 
                         'g', 'kg', 'ml', 'l', 'liter', 'liters', 'piece', 'pieces',
                         'pcs', 'of', 'medium', 'large', 'small', 'fresh', 'dried'}
        
        # Filter out numbers and quantity words
        ingredient_words = [w for w in words if not w.replace('.', '').isdigit() 
                          and w not in quantity_words]
        
        return ' '.join(ingredient_words) if ingredient_words else ingredient_text.lower()
    
    def is_ingredient_available(recipe_ingredient: str, available_list: List[str]) -> bool:
        """Check if ingredient is available in the pantry"""
        ingredient_name = extract_ingredient_name(recipe_ingredient)
        
        # Check for exact or partial matches
        for available in available_list:
            available_clean = extract_ingredient_name(available)
            
            # Check if main ingredient word is present
            ingredient_words = ingredient_name.split()
            available_words = available_clean.split()
            
            # If any significant word matches, consider it available
            for ing_word in ingredient_words:
                if len(ing_word) > 3:  # Ignore short words like 'of', 'in'
                    for avail_word in available_words:
                        if ing_word in avail_word or avail_word in ing_word:
                            return True
        
        return False
    
    shopping_list = []
    already_available = []
    
    for ingredient in recipe_ingredients:
        category = categorize_ingredient(ingredient)
        
        if is_ingredient_available(ingredient, available_list):
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
    
    # Group by category for easier shopping
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


def categorize_ingredient(ingredient: str) -> str:
    """Categorize ingredient for shopping organization"""
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


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class NutritionFacts(BaseModel):
    """Nutritional information per serving"""
    calories: float = Field(description="Total calories")
    protein_g: float = Field(description="Protein in grams")
    fat_total_g: float = Field(description="Total fat in grams")
    carbohydrates_total_g: float = Field(description="Total carbohydrates in grams")
    fiber_g: Optional[float] = Field(None, description="Fiber in grams")
    sugar_g: Optional[float] = Field(None, description="Sugar in grams")
    sodium_mg: Optional[float] = Field(None, description="Sodium in milligrams")


class ShoppingItem(BaseModel):
    """Individual shopping list item"""
    item: str = Field(description="Ingredient name with quantity")
    category: str = Field(description="Shopping category")
    status: str = Field(description="Status: need_to_purchase or available")


class RecipeOutput(BaseModel):
    """Final recipe output structure"""
    context: str = Field(default="https://schema.org", alias="@context")
    type: str = Field(default="Recipe", alias="@type")
    name: str = Field(description="Recipe name")
    description: str = Field(description="Brief description of the dish")
    author: str = Field(description="Recipe author/chef name")
    cuisine: str = Field(description="Cuisine type")
    difficulty: str = Field(description="Difficulty level: Easy, Medium, Hard")
    
    prepTime: str = Field(description="Prep time in ISO 8601 format (e.g., PT15M)")
    cookTime: str = Field(description="Cook time in ISO 8601 format (e.g., PT30M)")
    totalTime: str = Field(description="Total time in ISO 8601 format")
    
    recipeYield: str = Field(description="Number of servings")
    recipeIngredient: List[str] = Field(description="List of ingredients with quantities")
    recipeInstructions: List[str] = Field(description="Step-by-step cooking instructions")
    
    nutrition: NutritionFacts = Field(description="Nutritional information per serving")
    suitableForDiet: Optional[List[str]] = Field(None, description="Dietary restrictions met")
    
    # Shopping and planning
    shoppingList: List[ShoppingItem] = Field(description="Items that need to be purchased")
    availableItems: List[ShoppingItem] = Field(description="Items already available")
    shoppingByCategory: Union[Dict[str, List[str]], str] = Field(description="Shopping list organized by category")
    
    # Additional metadata
    keywords: List[str] = Field(description="Recipe keywords for searchability")
    datePublished: str = Field(description="Publication date")
    image: str = Field(default="https://via.placeholder.com/800x600", description="Recipe image URL")
    
    @field_validator('shoppingByCategory', mode='before')
    @classmethod
    def parse_shopping_by_category(cls, v):
        """Parse shoppingByCategory if it comes as a JSON string"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v if v is not None else {}


# =============================================================================
# STATE DEFINITION
# =============================================================================

class RecipeState(TypedDict):
    """State passed between agents"""
    # Input parameters
    available_ingredients: str
    dietary_restrictions: List[str]
    cuisine: str
    difficulty: str
    servings: int
    cooking_time_minutes: int
    
    # Agent outputs
    recipe_plan: Optional[str]
    nutrition_data: Optional[Dict]
    shopping_data: Optional[Dict]
    final_recipe: Optional[Dict]
    
    # Messages for conversation
    messages: Annotated[List, operator.add]
    
    # Tracking
    current_agent: str
    errors: Annotated[List[str], operator.add]


# =============================================================================
# LLM SETUP
# =============================================================================

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


# =============================================================================
# AGENT 1: PLANNING & NUTRITION AGENT
# =============================================================================

def planning_agent(state: RecipeState) -> RecipeState:
    """
    Agent 1: Creates recipe plan and fetches nutrition information
    """
    print("\n🔵 AGENT 1: Planning & Nutrition Agent")
    
    llm = get_llm(temperature=0.7, tools=[get_nutrition_info])
    
    system_message = """You are a professional chef and recipe planner.
Your tasks:
1. Analyze available ingredients and create a recipe concept
2. List the exact ingredients needed with quantities
3. Call the get_nutrition_info tool with the ingredients list
4. Create a basic recipe plan with cooking steps

Be specific with ingredient quantities and make sure they're realistic for the given servings."""
    
    user_message = f"""Create a recipe plan with these requirements:

Available Ingredients: {state['available_ingredients']}
Dietary Restrictions: {', '.join(state['dietary_restrictions'])}
Cuisine: {state['cuisine']}
Difficulty: {state['difficulty']}
Servings: {state['servings']}
Target Cooking Time: {state['cooking_time_minutes']} minutes

Please:
1. Design a recipe that uses available ingredients
2. List all ingredients with precise quantities
3. Use the get_nutrition_info tool to get nutritional data
4. Provide a step-by-step cooking plan"""
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message)
    ]
    
    # Get AI response with tool calls
    response = llm.invoke(messages)
    
    # Execute tool calls if present
    nutrition_data = None
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'get_nutrition_info':
                print(f"  📊 Fetching nutrition data...")
                result = get_nutrition_info.invoke(tool_call['args'])
                nutrition_data = result
                print(f"  ✅ Nutrition data retrieved: {result.get('success', False)}")
    
    state['recipe_plan'] = response.content
    state['nutrition_data'] = nutrition_data
    state['messages'].append(AIMessage(content=f"Planning Agent: {response.content}"))
    state['current_agent'] = 'planning'
    
    return state


# =============================================================================
# AGENT 2: SHOPPING & REQUIREMENTS AGENT
# =============================================================================

def shopping_agent(state: RecipeState) -> RecipeState:
    """
    Agent 2: Generates shopping list by comparing recipe vs available ingredients
    """
    print("\n🟢 AGENT 2: Shopping & Requirements Agent")
    
    llm = get_llm(temperature=0.5, tools=[compare_and_generate_shopping_list])
    
    system_message = """You are a shopping list expert and pantry management specialist.
Your tasks:
1. Extract ALL ingredients from the recipe plan with their exact quantities
2. Use the compare_and_generate_shopping_list tool to compare with available ingredients
3. Identify what needs to be purchased vs what's already available
4. Organize the shopping list by category for efficient shopping
5. Provide preparation tips and equipment needed

IMPORTANT: You MUST call the compare_and_generate_shopping_list tool with:
- recipe_ingredients: A LIST of strings, each containing one ingredient with quantity
- available_ingredients: The exact string of available ingredients provided

Be thorough and accurate in ingredient extraction."""
    
    user_message = f"""Based on this recipe plan, create a comprehensive shopping list:

Recipe Plan:
{state['recipe_plan']}

Available Ingredients in Pantry:
{state['available_ingredients']}

Dietary Restrictions: {', '.join(state['dietary_restrictions'])}

Step 1: Extract ALL ingredients with their exact quantities from the recipe plan above.
Step 2: Call the compare_and_generate_shopping_list tool with:
   - recipe_ingredients: ["ingredient 1 with quantity", "ingredient 2 with quantity", ...]
   - available_ingredients: "{state['available_ingredients']}"

Step 3: Based on the tool results, explain:
   - What needs to be purchased
   - What's already available
   - Items grouped by shopping category
   - Required cooking equipment
   - Any substitutions for dietary restrictions"""
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message)
    ]
    
    # Get response with tool calls
    response = llm.invoke(messages)
    
    # Debug: Print response details
    print(f"  📋 Response type: {type(response)}")
    print(f"  📋 Has tool_calls attr: {hasattr(response, 'tool_calls')}")
    if hasattr(response, 'tool_calls'):
        print(f"  📋 Tool calls: {response.tool_calls}")
    
    # Execute tool calls
    shopping_data = None
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"  📞 Found {len(response.tool_calls)} tool call(s)")
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'compare_and_generate_shopping_list':
                print(f"  🛒 Comparing ingredients with pantry...")
                try:
                    result = compare_and_generate_shopping_list.invoke(tool_call['args'])
                    shopping_data = result
                    print(f"  ✅ Shopping list generated:")
                    print(f"     Items to buy: {result.get('items_to_buy', 0)}")
                    print(f"     Items available: {result.get('items_in_stock', 0)}")
                except Exception as e:
                    print(f"  ❌ Tool execution error: {e}")
                    state['errors'].append(f"Shopping tool error: {str(e)}")
    else:
        print(f"  ⚠️  No tool calls found in response")
        state['errors'].append("Shopping agent did not call comparison tool")
    
    # Fallback: Create basic shopping data if tool wasn't called
    if shopping_data is None:
        print(f"  ⚠️  Creating fallback shopping data...")
        shopping_data = {
            "items_to_purchase": [],
            "items_available": [],
            "shopping_by_category": {},
            "total_items_needed": 0,
            "items_to_buy": 0,
            "items_in_stock": 0
        }
    
    state['shopping_data'] = shopping_data
    state['messages'].append(AIMessage(content=f"Shopping Agent: {response.content}"))
    state['current_agent'] = 'shopping'
    
    return state


# =============================================================================
# AGENT 3: STRUCTURING AGENT
# =============================================================================

def structuring_agent(state: RecipeState) -> RecipeState:
    """
    Agent 3: Formats everything into final recipe schema
    """
    print("\n🟡 AGENT 3: Structuring Agent")
    
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(RecipeOutput)
    
    # Calculate per-serving nutrition
    nutrition_data = state.get('nutrition_data', {})
    if nutrition_data and nutrition_data.get('success'):
        total_nutrition = nutrition_data['nutrition']
        servings = state['servings']
        per_serving_nutrition = {
            key: round(value / servings, 2)
            for key, value in total_nutrition.items()
        }
    else:
        # Default values if nutrition fetch failed
        per_serving_nutrition = {
            "calories": 350.0,
            "protein_g": 15.0,
            "fat_total_g": 12.0,
            "carbohydrates_total_g": 45.0,
            "fiber_g": 5.0,
            "sugar_g": 3.0,
            "sodium_mg": 400.0
        }
    
    # Get shopping list data with safe defaults
    shopping_data = state.get('shopping_data') or {}
    items_to_purchase = shopping_data.get('items_to_purchase', [])
    items_available = shopping_data.get('items_available', [])
    shopping_by_category = shopping_data.get('shopping_by_category', {})
    
    # Map dietary restrictions to schema.org URLs
    diet_mapping = {
        "Vegetarian": "https://schema.org/VegetarianDiet",
        "Vegan": "https://schema.org/VeganDiet",
        "Gluten-free": "https://schema.org/GlutenFreeDiet",
        "Diabetic": "https://schema.org/DiabeticDiet",
        "Halal": "https://schema.org/HalalDiet",
        "Kosher": "https://schema.org/KosherDiet",
        "Low-calorie": "https://schema.org/LowCalorieDiet",
        "Low-fat": "https://schema.org/LowFatDiet",
        "Low-sodium": "https://schema.org/LowSaltDiet"
    }
    
    suitable_diets = [
        diet_mapping.get(restriction, restriction)
        for restriction in state['dietary_restrictions']
        if restriction in diet_mapping
    ]
    
    system_message = """You are a recipe documentation specialist.
Create a complete, properly structured recipe following the schema.org Recipe format.

IMPORTANT:
- Use ISO 8601 duration format for times (PT15M for 15 minutes, PT1H30M for 1.5 hours)
- Ensure all ingredients have quantities
- Instructions must be clear, numbered steps
- Include all provided nutrition and shopping data
- Generate appropriate recipe keywords

CRITICAL FORMAT REQUIREMENTS:
- shoppingByCategory MUST be a dictionary/object, NOT a string
  Correct: {"Produce": ["item1", "item2"], "Dairy": ["item3"]}
  Wrong: '{"Produce": ["item1"]}'  (string format is wrong)
- shoppingList and availableItems must be arrays of objects with item, category, status fields
- All lists must be actual arrays, not strings"""
    
    user_message = f"""Create the final structured recipe:

RECIPE PLAN:
{state['recipe_plan']}

NUTRITION (per serving):
{json.dumps(per_serving_nutrition, indent=2)}

SHOPPING DATA:
Items to Purchase (as JSON array):
{json.dumps(items_to_purchase, indent=2)}

Items Already Available (as JSON array):
{json.dumps(items_available, indent=2)}

Shopping by Category (as JSON object - DO NOT stringify this):
{json.dumps(shopping_by_category, indent=2)}

IMPORTANT: When creating the output, shoppingByCategory should be a dictionary object like:
{{"Produce": ["item1", "item2"], "Dairy": ["item3"]}}
NOT a string. Use the data above directly as an object.

REQUIREMENTS:
- Cuisine: {state['cuisine']}
- Difficulty: {state['difficulty']}
- Servings: {state['servings']}
- Cooking Time: {state['cooking_time_minutes']} minutes
- Dietary Restrictions: {', '.join(state['dietary_restrictions'])}

Generate a complete recipe with:
1. Creative, descriptive recipe name
2. Appealing description
3. All ingredients with exact quantities
4. Clear step-by-step instructions
5. Proper time formats (ISO 8601)
6. Current date for datePublished (format: YYYY-MM-DD)
7. Relevant keywords for search
"""
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message)
    ]
    
    try:
        # Get structured output
        recipe_output = structured_llm.invoke(messages)
        
        # Convert to dict and add computed fields
        recipe_dict = recipe_output.model_dump(by_alias=True)
        
        # Ensure shopping list data is included
        recipe_dict['shoppingList'] = items_to_purchase
        recipe_dict['availableItems'] = items_available
        recipe_dict['shoppingByCategory'] = shopping_by_category
        
        # Add suitable diets
        if suitable_diets:
            recipe_dict['suitableForDiet'] = suitable_diets
        
        state['final_recipe'] = recipe_dict
        print(f"  ✅ Recipe structured: {recipe_dict['name']}")
        
    except Exception as e:
        print(f"  ❌ Error structuring recipe: {e}")
        state['errors'].append(f"Structuring error: {str(e)}")
    
    state['current_agent'] = 'structuring'
    return state


# =============================================================================
# BUILD LANGGRAPH WORKFLOW
# =============================================================================

def build_recipe_graph():
    """Build the LangGraph workflow"""
    
    workflow = StateGraph(RecipeState)
    
    # Add nodes (agents)
    workflow.add_node("planning", planning_agent)
    workflow.add_node("shopping", shopping_agent)
    workflow.add_node("structuring", structuring_agent)
    
    # Define edges (workflow)
    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "shopping")
    workflow.add_edge("shopping", "structuring")
    workflow.add_edge("structuring", END)
    
    return workflow.compile()


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def generate_recipe(
    available_ingredients: str,
    dietary_restrictions: List[str],
    cuisine: str = "South Indian",
    difficulty: str = "Medium",
    servings: int = 2,
    cooking_time_minutes: int = 50
) -> Dict:
    """
    Main function to generate recipe using multi-agent system
    
    Args:
        available_ingredients: Comma-separated list of available ingredients
        dietary_restrictions: List of dietary restrictions
        cuisine: Type of cuisine
        difficulty: Recipe difficulty level
        servings: Number of servings
        cooking_time_minutes: Target cooking time
    
    Returns:
        Dictionary with final recipe in schema.org format
    """
    
    print("=" * 80)
    print("🍳 MULTI-AGENT RECIPE GENERATION SYSTEM")
    print("=" * 80)
    
    # Initialize state
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
    
    # Build and run graph
    graph = build_recipe_graph()
    
    print("\n🚀 Starting multi-agent workflow...\n")
    
    # Execute workflow
    final_state = graph.invoke(initial_state)
    
    print("\n" + "=" * 80)
    print("✨ RECIPE GENERATION COMPLETE")
    print("=" * 80)
    
    if final_state.get('errors'):
        print("\n⚠️  Warnings/Errors:")
        for error in final_state['errors']:
            print(f"  - {error}")
    
    return final_state['final_recipe']


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example ingredients from your document
    ingredients = """1 kg of Rice, 2 kg of Wheat Flour, 1 kg of Sugar, 500 g of Salt, 
    1 L of Sunflower Oil, 500 g of Pasta, 250 g of Butter, 12 pcs of Eggs, 1 L of Milk, 
    1 kg of Potatoes, 500 g of Onions, 200 g of Garlic, 500 g of Tomatoes, 
    100 g of Black Pepper, 100 g of Turmeric, 100 g of Cumin Seeds"""
    
    dietary_restrictions = ["Vegetarian", "Gluten-free"]
    
    # Generate recipe
    recipe = generate_recipe(
        available_ingredients=ingredients,
        dietary_restrictions=dietary_restrictions,
        cuisine="South Indian",
        difficulty="Medium",
        servings=4,
        cooking_time_minutes=45
    )
    
    # Save to file
    if recipe:
        output_file = "recipe_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(recipe, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Recipe saved to: {output_file}")
        print(f"\n📋 Recipe: {recipe['name']}")
        print(f"🍽️  Servings: {recipe['recipeYield']}")
        print(f"⏱️  Total Time: {recipe['totalTime']}")
        print(f"🛒 Items to buy: {len(recipe.get('shoppingList', []))}")
        print(f"✅ Items available: {len(recipe.get('availableItems', []))}")
        print(f"🔥 Calories per serving: {recipe['nutrition']['calories']}")
    else:
        print("\n❌ Recipe generation failed!")