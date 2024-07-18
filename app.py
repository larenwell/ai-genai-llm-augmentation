## These are reference imports, please change with anything you want to use
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import streamlit as st


## Function to get response from the model
def getAgentResponse(input_text, no_words, category):
    llm = CTransformers(model = '',
                        model_type = '',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    
    ## PromptTemplate
    template = """Write a  {category} on {input_text} in less than {no_words} words"""

    prompt = PromptTemplate(input_variables = ["input_text", "no_words", "category"],
                            template = template)
    
    ## Generate the reponse
    response = llm(prompt.format(category=category,input_text=input_text,no_words=no_words))
    print(response)
    return response