#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st 
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import ArxivAPIWrapper
from langchain.tools import YouTubeSearchTool
from langchain.utilities import SerpAPIWrapper
import os
openai_key = os.environ.get("OPENAI_API_KEY")
serpapi_key = os.environ.get("SERPAPI_API_KEY")

# App framework
st.title('Trend Predictor For Writing Papers🦜🔗 ')
prompt = st.text_input('Enter your prompt here') 

# Prompt templates

title_template=PromptTemplate(
    input_variables=["topic",'search'], 
    template="""
    You are a talented researcher, your job is to research the latest trends concerning {topic}. 
    When you answer, follow this format: 

    A) Briefly explain the topic while leveraging this search:{search} \n
    B) State your predictions \n
    """
)
agenda_template = PromptTemplate(
    input_variables = ['text', 'arxiv'], 
    template='List a table of content for a new research paper concerning this topic: {text} while leveraging this paper research:{arxiv} '
)
ytb_template = PromptTemplate(
    input_variables = ['script', 'ytb'], 
    template='Recommend me helpful youtube video regarding this topic {script} while leveraging this youtube research:{ytb} '
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
agenda_memory = ConversationBufferMemory(input_key='text', memory_key='chat_history')
ytb_memory = ConversationBufferMemory(input_key='script', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.3) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='text', memory=title_memory)
agenda_chain = LLMChain(llm=llm, prompt=agenda_template, verbose=True, output_key='script', memory=agenda_memory)
ytb_chain = LLMChain(llm=llm, prompt=ytb_template, verbose=True, output_key='youtube', memory=ytb_memory)


#Tools
ytb= YouTubeSearchTool()
arxiv = ArxivAPIWrapper()
search = SerpAPIWrapper()

# Show stuff to the screen 
if prompt: 
    serp_research=search.run(prompt)
    text = title_chain.predict_and_parse(topic=prompt, search=serp_research)
    arxiv_research = arxiv.run(prompt) 
    script = agenda_chain.predict_and_parse(text=text, arxiv=arxiv_research)
    ytb_research=ytb.run(prompt)
    youtube = ytb_chain.predict_and_parse(script=script, ytb=ytb)

    st.write(text) 
    st.write(script) 
    st.write(youtube)

    with st.expander('Predictions History'): 
        st.info(title_memory.buffer)

    with st.expander('Predictions History'): 
        st.info(agenda_memory.buffer)

    with st.expander('Paper Research'): 
        st.info(arxiv_research)

    with st.expander('Youtube Research'): 
        st.info(ytb_research)



