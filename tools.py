from langchain_community.tools import YouTubeSearchTool
from langchain.tools import Tool
from langchain.vectorstores import Chroma
# from langchain_community.rag import retrieve_from_chroma

# Simulated retrieve_from_chroma function
def retrieve_from_chroma(dish_name, vectorstore):
    return [
        type('Doc', (object,), {'page_content': f"Document content for {dish_name}"})
    ]

# Simulated recipe generation
def generate_recipe(dish_name):
    if not dish_name.strip():
        return {"error": "Dish name is empty"}
    return {
        "dish_name": dish_name,
        "ingredients": ["ingredient 1", "ingredient 2", "ingredient 3"],
        "steps": ["Step 1", "Step 2", "Step 3"]
    }

class YouTubeTool:
    def __init__(self):
        self.tool = YouTubeSearchTool()

    def get_youtube_video(self, dish_name):
        try:
            result = self.tool.run(dish_name + " recipe")
            return result or {"error": "No video found for this recipe"}
        except Exception as e:
            return {"error": f"Failed to fetch video: {str(e)}"}

def rag_retriever_tool(vector):
    """
    RAG retrieval tool for relevant documents.

    Args:
        vector (object): The vector store instance.

    Returns:
        Tool: RAG retriever tool.
    """
    return Tool(
        name="RAG Retriever",
        func=lambda dish_name: "\n\n".join(doc.page_content for doc in retrieve_from_chroma(dish_name, vectorstore=vector)),
        description="Retrieves relevant documents for a given topic using a vector store."
    )

