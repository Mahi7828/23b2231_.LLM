import streamlit as st
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os
from IPython.display import Markdown
import textwrap

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

model = genai.GenerativeModel("gemini-pro")


def to_markdown(text):
  #text=text.replace('.' , '*')
  return (textwrap.indent(text , '> ' , predicate=lambda _:True))



def getgeminiresponse(input_text , no_words, blog_style):
    chat = model.start_chat

    template="""
        write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.


        """
    
    prompt = PromptTemplate(input_variables=["blog_style" , "input_text" , "no_words"],template=template)
    
    ## generate the response from the gemini pro  model

    response = model.generate_content((prompt.format(blog_style= blog_style , input_text = input_text , no_words= no_words)))
    markdown_text = to_markdown(response.text)
    return markdown_text



st.set_page_config(page_title="Generate Blogs" , page_icon = "🤖",
                   layout="centered",
                   initial_sidebar_state="collapsed")

st.header("Generate Blogs - The way you Want")

input_text=st.text_input("Enter the Blog Topic")

#creating two more coloumns for additional fields

col1 , col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("No of words")

with col2:
    blog_style=st.selectbox("Main Readers of the Blog" , ('Researchers' , 'Data Scientists' , 'Professional' , 'Students' , 'Common People') , index = 0 )


submit=st.button("Generate")
##final response

if submit:
    st.markdown(getgeminiresponse(input_text, no_words, blog_style))





    

    


    
    
