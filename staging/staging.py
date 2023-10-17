import gradio as gr
import random
import time
import os
import json
import sagemaker
import logging

from langchain.vectorstores import Vectara
from langchain.vectorstores.vectara import VectaraRetriever
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain, ConversationChain, LLMChain, LLMCheckerChain

from typing import Dict

from langchain import PromptTemplate
from langchain.llms.sagemaker_endpoint import LLMContentHandler

# parts of a model: chat, bot
# - chat is the actualy chat history / output on the screen
# - bot calls the llm endpoint with some prompts and context and langchain magic

from abc import abstractmethod
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
import boto3, time, os, uuid
from botocore.exceptions import ClientError

from langchain.agents import AgentExecutor, AgentType, initialize_agent, Tool, ZeroShotAgent
from langchain.llms import OpenAI
from langchain.prompts import MessagesPlaceholder

import langchain

langchain.debug = True

from serpapi import GoogleSearch

GoogleSearch.SERP_API_KEY = "5567e356a3e19133465bc68755a124268543a7dd0b2809d75b038797b43626ab"

def filtered_search(results):
    new_dict = {}
    if('sports_results' in results):
        new_dict['sports_results'] = results['sports_results']
    if('organic_results' in results):
        new_dict['organic_results'] = results['organic_results']
    return new_dict

def gov_search(q):
    return filtered_search(GoogleSearch({
        'q': "site:*.gov " + q,
        'num': 5
        }).get_dict())

def general_search(q):
    return filtered_search(GoogleSearch({
        'q': q,
        'num': 5
        }).get_dict())


# system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
# too much safety, hurts accuracy

with gr.Blocks(title="OpenProBono",
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.indigo, 
        secondary_hue=gr.themes.colors.blue,
        font=gr.themes.GoogleFont("Open Sans"),
        radius_size=gr.themes.sizes.radius_lg),
        # .set(
        #     button_primary_background_fill="*primary_200",
        #     button_primary_background_fill_hover="*primary_300",
        # ),
    css="footer {visibility: hidden}"
    ) as demo:

    gpt3_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613')

    def print_email(email):
        print(email)
        print("^^ this is the email ^^")
        return email
    
    def add_text(history, text):
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)


    def add_file(history, file):
        loader = TextLoader(file.name)
        file_text = loader.load()

        history = history + [((file.name,), None)]
        return history


    def openai_bot(history):
        tools = [
            Tool(
                name="search",
                func=general_search,
                description="useful for when you need to answer questions about current events. You should ask targeted questions. Always cite your sources.",
            ),
            Tool(
                name="government-search",
                func=gov_search,
                description="useful for when you need to answer questions or find resources about government and laws. Always cite your sources.",
            )
        ]
        history_langchain_format = ChatMessageHistory()
        for i in range(0, len(history)-1):
            (human, ai) = history[i]
            history_langchain_format.add_user_message(human)
            history_langchain_format.add_ai_message(ai)
        memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")

        system_message = 'You are a helpful AI assistant. '
        #system_message += user_prompt
        system_message += '. ALWAYS return a "SOURCES" part in your answer.'
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }
        agent = initialize_agent(
            tools=tools,
            llm=gpt3_llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )
        agent.agent.prompt.messages[0].content = system_message
        print(agent.agent.prompt)
        print("^^ this agent here^^")
        bot_message = agent.run(history[-1][0])
        history[-1][1] = bot_message
        yield history
    


    gr.Markdown("OpenProBono")
    with gr.Row():
        openai_chat = gr.Chatbot(
            [],
            elem_id="chat",
            label="OpenProBono",
            show_label=True,
            #bubble_full_width=True,
            #avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            label="input",
            show_label=False,
            placeholder="Enter query", #, or upload an image",
            container=False,
        )
        subbtn = gr.Button("Submit", variant="primary")
        clearopenai = gr.ClearButton([txt, openai_chat])
        #btn = gr.UploadButton("📁", file_types=["text"])


    # file_msg = btn.upload(add_file, [openai, btn], [openai], queue=False).then(
    #     bot, openai, openai
    # )
    with gr.Accordion("Details"):
        with gr.Row():
            emailtxt = gr.Textbox(
                scale=4,
                label="input",
                show_label=False,
                placeholder="Enter your email to sign up for updates", #, or upload an image",
                container=False,
            )
            emailbtn = gr.Button("Submit")
        gr.Markdown("This demo is a beta meant for informational purposes, demonstrating the abilities of our current technology and to compare different variations of models, prompting methods, document upload, and other features as we continually improve. The data sent in the demo is not guaranteed to be kept private. We will keep iterating on this demo, so keep an eye out for frequent updates. This is not legal advice. Learn more at www.openprobono.com.")

    txt_msg = txt.submit(add_text, [openai_chat, txt], [openai_chat, txt], queue=False).then(
        openai_bot, [openai_chat], openai_chat
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    sub_msg = subbtn.click(add_text, [openai_chat, txt], [openai_chat, txt], queue=False, api_name="submit").then(
        openai_bot, [openai_chat], openai_chat
    )
    sub_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    email_txt = emailtxt.submit(print_email, [emailtxt], [emailtxt], queue=False)
    email_msg = emailbtn.click(print_email, [emailtxt], [emailtxt], queue=False)
    
demo.queue()

demo.launch(root_path="/staging",server_port=7863,favicon_path="./missing.ico")
