import streamlit as st
import chains
import vectordb
import agents

def recipe_generator_app():
    """
    Generates Recipe Generator App with Streamlit, providing user input and displaying output.
    Includes a sidebar with two sections: Recipe Generator and File Ingestion for RAG.
    """

    # Sidebar configuration
    st.sidebar.title("Menu")
    section = st.sidebar.radio(
        "Choose a section:",
        ("Recipe Generator RAG", "RAG File Ingestion")
    )

    # DB initialization
    vectordatabase = vectordb.initialize_chroma()

    # Recipe generation page
    if section == "Recipe Generator RAG":
        st.title("Let's generate a recipe! 🍲")

        with st.form("recipe_generator"):
            dish_name = st.text_input("Enter a dish name for the recipe:")
        
            submitted = st.form_submit_button("Submit")

            is_rag_enabled = st.checkbox("Enable RAG")
            is_agent_enabled = st.checkbox("Enable Agent")

            if submitted:
                if is_rag_enabled and is_agent_enabled:
                    response = agents.generate_recipe_with_rag_agent(dish_name, vectordatabase)
                elif is_agent_enabled:
                    response = agents.generate_recipe_with_agent(dish_name)
                elif is_rag_enabled:
                    response = chains.generate_recipe_rag_chain(dish_name,  vectordatabase)
                else:
                    response = chains.generate_recipe_chain(dish_name)

                st.info(response)

    # File ingestion page
    elif section == "RAG File Ingestion":
        st.title("RAG File Ingestion")

        uploaded_file = st.file_uploader("Upload a file:", type=["txt", "csv", "docx", "pdf"])

        if uploaded_file is not None:
            vectordb.store_pdf_in_chroma(uploaded_file, vectordatabase)
            st.success(f"File '{uploaded_file.name}' uploaded and file embedding stored successfully!")

recipe_generator_app()
