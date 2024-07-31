import asyncio
import streamlit as st
from retrieval import docs_retriever, evaluate_summary

# Set the title of the Streamlit application
st.title("LLM Augmentation with Project Gutenberg Catalog")

# Write a description and instructions for the tool
st.write("""
Welcome to the LLM Augmentation with Project Gutenberg Catalog! This tool allows you to generate detailed summaries of texts from the Project Gutenberg catalog. 

**Instructions:**
1. **Enter your question**: Provide a specific question or request for a summary related to the texts in the Project Gutenberg catalog.
2. **Reference summary (optional)**: If you have a reference summary for evaluation, you can enter it in the provided text area.
3. **Submit**: Click the submit button to generate the summary and, if applicable, evaluate it against the reference summary.
""")

# Create a form for user input
with st.form("chat-form"):
    # Text input for the user's question
    prompt = st.text_input("Enter your question", placeholder="Provide a detailed summary of The Souls of Black Folk")
    
    # Text area for the reference summary (optional)
    reference_summary = st.text_area("Enter the reference summary for evaluation (optional)", placeholder="Enter reference summary here...")
    
    # Submit button for the form
    submit_button = st.form_submit_button("Submit")
    
    # If the submit button is clicked, process the input
    if submit_button:
        # Retrieve documents and generate a summary asynchronously
        result = asyncio.run(docs_retriever(prompt))
        
        # Display the generated summary
        st.write("Generated Summary:")
        st.write(result)
        
        # If a reference summary is provided, evaluate the generated summary
        if reference_summary:
            metrics = evaluate_summary(result, reference_summary)
            st.write("Evaluation Metrics:")
            st.json(metrics)
