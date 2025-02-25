from langchain_core.output_parsers import StrOutputParser
import models
import prompts
import vectordb

#### GENERATION ####
def generate_recipe_chain(dish_name):
    """
    Generate Recipe using basic prompt LLM chain

    Args:
        dish_name: Name of the dish to generate a recipe for

    Returns:
        str: Generated recipe content
    """
        
    llm = models.create_chat_groq_model()

    prompt_template = prompts.recipe_generator_prompt()
    # prompt_template = prompts.recipe_generator_prompt_from_hub()

    # Creating chain
    chain = prompt_template | llm

    response = chain.invoke({
        "dish_name": dish_name  # Pass dish_name into prompt correctly
    })

    return response.content


#### RETRIEVAL and GENERATION ####
def generate_recipe_rag_chain(dish_name, vector):
    """
    Creates a RAG chain for retrieval and generation.

    Args:
        dish_name: Name of the dish for recipe generation
        vector: Instance of vector store

    Returns:
        response: Generated recipe content from RAG chain
    """
    # Prompt for RAG model
    prompt = prompts.recipe_generator_agent()

    # Initialize LLM model
    llm = models.create_chat_groq_model()

    # Post-processing function for documents
    def format_docs(docs):
        # Ensure docs are formatted as a string for the model
        return "\n\n".join([doc.page_content for doc in docs])

    # Retrieve relevant documents from the vector store
    retriever = vectordb.retrieve_from_chroma(dish_name, vectorstore=vector)
    
    # If no documents are found, handle gracefully
    if not retriever:
        return "No relevant information found for the recipe."

    # Create RAG chain
    rag_chain = prompt | llm | StrOutputParser()

    # Invoke the RAG chain
    response = rag_chain.invoke({
        "context": format_docs(retriever),
        "dish_name": dish_name  # Make sure this matches the expected input in the prompt
    })

    return response