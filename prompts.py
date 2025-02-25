from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain import hub


#This is for normal chatbot for creating cooking recipes
def recipe_generator_prompt():
    """
    Generates Prompt template from the system and user messages

    Returns:
        ChatPromptTemplate -> Configured ChatPromptTemplate instance
    """

    system_msg = '''
               You are an AI-powered culinary assistant, Create a unique and delicious recipe based on a given dish or cuisine. Generate a list of ingredients along with precise measurements and step-by-step cooking instructions. Ensure the recipe is well-structured, easy to follow, and includes variations for dietary preferences if applicable. The output should be formatted as follows:"

                    Recipe Name: (Generated based on the input or AI suggestion)
                    Ingredients:
                     List each ingredient with accurate measurements (e.g., "2 cups of flour," "1 tsp of salt")
                    Optionally include substitutes for dietary restrictions (e.g., "Almond milk instead of regular milk for a vegan option")
                    Instructions:
                    Provide clear, numbered, step-by-step cooking directions.
                    Mention cooking times, temperatures, and techniques (e.g., "Preheat oven to 375°F").
                    Keep the steps concise yet informative.
                    Additional Features:
                    Suggest serving size and estimated preparation/cooking time.
                    Offer variations (e.g., “For a spicier version, add extra chili flakes”).
                    Recommend ideal side dishes or pairings.
                ''' 
    
    user_msg = "Please provide the Name of the Dish {dish_name} you'd like to create"
    
    
    prompt_template = ChatPromptTemplate([
        ("system", system_msg),
        ("user", user_msg)
    ])
    return prompt_template
#hub pull for normal chatbot for creating cooking recipes

# def poem_generator_prompt_from_hub(template="vethika-poem-generator/recipe_generator"):
#     """
#     Generates Prompt template from the LangSmith prompt hub
#     prompt = hub.pull("vethika-poem-generator/recipe_generator")

#     Returns:
#         ChatPromptTemplate -> ChatPromptTemplate instance pulled from LangSmith Hub
#     """
    
#     prompt_template = hub.pull(template)
#     return prompt_template

#This is for chatbot using rag  for creating cooking recipes

    
def recipe_generator_rag_prompt():
    """
    Generates a RAG-enabled Prompt template for recipe generation.

    Returns:
        ChatPromptTemplate -> Configured ChatPromptTemplate instance
    """

    system_msg = '''
               You are an AI-powered culinary assistant, Create a unique and delicious recipe based on a given dish or cuisine. Generate a list of ingredients along with precise measurements and step-by-step cooking instructions. Ensure the recipe is well-structured, easy to follow, and includes variations for dietary preferences if applicable. The output should be formatted as follows:"

                    Recipe Name: (Generated based on the input or AI suggestion)
                    Ingredients:
                     List each ingredient with accurate measurements (e.g., "2 cups of flour," "1 tsp of salt")
                    Optionally include substitutes for dietary restrictions (e.g., "Almond milk instead of regular milk for a vegan option")
                    Instructions:
                    Provide clear, numbered, step-by-step cooking directions.
                    Mention cooking times, temperatures, and techniques (e.g., "Preheat oven to 375°F").
                    Keep the steps concise yet informative.
                    Additional Features:
                    Suggest serving size and estimated preparation/cooking time.
                    Offer variations (e.g., “For a spicier version, add extra chili flakes”).
                    Recommend ideal side dishes or pairings.
                ''' 
    
    user_msg = "Please provide the Name of the Dish {dish_name} you'd like to create"

    prompt_template = ChatPromptTemplate([
        ("system", system_msg),
        ("user", user_msg)
    ])

    return prompt_template

    #This is  hub for chatbot using rag  for creating cooking recipes


def recipe_generator_rag_prompt_from_hub(template="vethika-poem-generator/recipe_generator"):
    """
    Generates Prompt template from the LangSmith prompt hub
    prompt = hub.pull("vethika-poem-generator/recipe_generator")

    Returns:
        ChatPromptTemplate -> ChatPromptTemplate instance pulled from LangSmith Hub
    """
    
    prompt_template = hub.pull(template)
    return prompt_template

    #This is  agent for chatbot   for creating cooking recipes

def recipe_generator_agent():
    """
    Generates Prompt template from the system and user messages

    Returns:
        ChatPromptTemplate -> Configured ChatPromptTemplate instance
    """

    system_msg = '''
               You are a dedicated recipe generator agent, specialized in crafting unique and delicious recipes based on a given dish or cuisine. Answer the following questions as best you can. You have access to the following tools:
                {tools}

                    Use the following format:

                Question: The input question you must answer.

                Thought: You should always think about what to do with the following restrictions:

                Only respond to queries explicitly requesting a recipe, ingredient list, or cooking instructions.
                The output must strictly be the recipe itself, formatted with a clear structure, including the recipe name, ingredients with precise measurements, step-by-step instructions, and optional variations for dietary preferences. No additional explanations, descriptions, or headers should be included.
                If the query is unrelated to recipe generation (e.g., generating poems, code, general knowledge questions, or any other non-recipe tasks), respond with:
                "I am a recipe generator agent, specialized in creating structured and accurate recipes. Please ask me a cooking-related query."
                Do not perform any tasks beyond recipe generation. Always fall back to the above message for non-recipe-related queries.
                Action: The action to take, should be one of [{tool_names}].
                Action Input: The input to the action.
                Observation: The result of the action.
                ...(this Thought/Action/Action Input/Observation can repeat for a maximum of N times)
                Thought: I now know the final answer.
                Final Answer: The final answer to the original input question.
                Begin!
                Question: {dish_name}
                Thought: {agent_scratchpad}
                '''
    
    
    prompt = PromptTemplate(
        input_variables=["dish_name", "tool_names", "agent_scratchpad","tools"],
        template=system_msg
    )
    return prompt

   #This is  agent hub for chatbot using rag  for creating cooking recipes 

def recipe_generator_agent_from_hub(template="vethika-poem-generator/recipe_generator_agent"):
    """
    Generates an template for agent to generate poem from the LangSmith hub.

    Returns:
        ChatPromptTemplate -> ChatPromptTemplate pulled from LangSmith Hub
    """
    agent = hub.pull(template, object_type="agent")
    return agent

   #This is   hub for chatbot using rag and agent  for creating cooking recipes

def recipe_generator_agent_with_rag_from_hub(template="vethika-poem-generator/recipe_generator_agent"):
    """
    Generates an template for agent to generate poem from the LangSmith hub.

    Returns:
        ChatPromptTemplate -> ChatPromptTemplate pulled from LangSmith Hub
    """
    agent = hub.pull(template, object_type="agent")
    return agent