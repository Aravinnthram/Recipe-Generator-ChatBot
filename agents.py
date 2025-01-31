from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import tool
import models
import prompts
import tools

def generate_recipe_with_agent(dish_name):
    try:
        print("dish_name1",dish_name)
        tools_list = [tools.generate_recipe(dish_name), tools.rag_retriever_tool(dish_name)]
        prompt_template = prompts.recipe_generator_prompt()
        llm = models.create_chat_groq_model()

        agent = create_react_agent(tools=tools_list, llm=llm, prompt=prompt_template)
        agent_executor = AgentExecutor(agent=agent, tools=tools_list)

        response = agent_executor.invoke({"dish_name": dish_name})
        return response
    except Exception as e:
        return {"error": f"Failed to generate recipe: {str(e)}"}

def generate_recipe_with_rag_agent(dish_name, vector):
    try:
        print("dish_name",dish_name)
        tools_list = [tool.rag_retriever_tool(vector), tool.generate_recipe(dish_name)]
        prompt_template = prompts.recipe_generator_rag_prompt()
        llm = models.create_chat_groq_model()

        agent = create_react_agent(tools=tools_list, llm=llm, prompt=prompt_template)
        agent_executor = AgentExecutor(agent=agent, tools=tools_list)

        response = agent_executor.invoke({"dish_name": dish_name})
        return response
    except Exception as e:
        return {"error": f"Failed to generate RAG recipe: {str(e)}"}
