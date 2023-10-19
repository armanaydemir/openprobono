import gradio as gr
import random
import time
import os
import json
import sagemaker

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

from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentExecutor, AgentType, initialize_agent, Tool, ZeroShotAgent
from langchain.llms import OpenAI
from langchain.prompts import MessagesPlaceholder

from langchain.chains.llm_symbolic_math.base import LLMSymbolicMathChain

import langchain

langchain.debug = True

search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions.",
    )
]


system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
# too much safety, hurts accuracy

with gr.Blocks(title="MathLLM",
    #font=gr.themes.GoogleFont("Open Sans"),
    css="footer {visibility: hidden}"
    )  as demo:


    gpt3_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613')
    
    def add_text(history, text):
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)


    def add_file(history, file):
        loader = TextLoader(file.name)
        file_text = loader.load()

        history = history + [((file.name,), None)]
        return history


    def math_bot(history):
        history_langchain_format = ChatMessageHistory()
        for i in range(0, len(history)-1):
            (human, ai) = history[i]
            history_langchain_format.add_user_message(human)
            history_langchain_format.add_ai_message(ai)
        memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")
        

        llm_math = LLMSymbolicMathChain.from_llm(
            llm=gpt3_llm, 
            memory=memory,
            verbose=True)
        bot_message = llm_math.run(history[-1][0])
        history[-1][1] = bot_message
        yield history

    with gr.Row():
        openai_chat = gr.Chatbot(
            [],
            elem_id="chat",
            label="Math Chat",
            show_label=True,
            #bubble_full_width=True,
            #avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        )

    # with gr.Row():
    #     contxt = gr.Textbox(
    #         scale=4,
    #         show_label=False,
    #         placeholder="Enter any context you want the AI to reference", #, or upload an image",
    #         container=False,
    #     )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            label="input",
            show_label=False,
            placeholder="Enter query", #, or upload an image",
            container=False,
        )
        subbtn = gr.Button("Submit")
        clearopenai = gr.ClearButton([txt, openai_chat])
        #btn = gr.UploadButton("📁", file_types=["text"])


    # file_msg = btn.upload(add_file, [openai, btn], [openai], queue=False).then(
    #     bot, openai, openai
    # )
    with gr.Accordion("Details"):
        gr.Markdown("This demo is a beta meant for informational purposes, demonstrating the abilities of our current technology and to compare different variations of models, prompting methods, document upload, and other features as we continually improve. The data sent in the demo is not guaranteed to be kept private. We will keep iterating on this demo, so keep an eye out for frequent updates. This is not legal advice.")

    txt_msg = txt.submit(add_text, [openai_chat, txt], [openai_chat, txt], queue=False).then(
        math_bot, [openai_chat], openai_chat
    )

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    sub_msg = subbtn.click(add_text, [openai_chat, txt], [openai_chat, txt], queue=False, api_name="submit").then(
        math_bot, [openai_chat], openai_chat
    )

    sub_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    
demo.queue()

demo.launch(root_path="/math",server_port=7862,favicon_path="./missing.ico")
