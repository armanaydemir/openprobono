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



with gr.Blocks(title="OpenProBono",
    #font=gr.themes.GoogleFont("Open Sans"),
    css="footer {visibility: hidden}") as demo:

    gr.HTML("""
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-MKDNM9G2PQ"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', 'G-MKDNM9G2PQ');
        </script>
        """)
    gpt3_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613')
    
    def add_text(history, text):
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)


    def add_file(history, file):
        loader = TextLoader(file.name)
        file_text = loader.load()

        history = history + [((file.name,), None)]
        return history


    def openai_bot(history):
        history_langchain_format = ChatMessageHistory()
        for i in range(0, len(history)-1):
            (human, ai) = history[i]
            history_langchain_format.add_user_message(human)
            history_langchain_format.add_ai_message(ai)
        memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")
        

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
        bot_message = agent.run(history[-1][0])
        history[-1][1] = bot_message
        yield history

    # def openai_bot_no_search(history):
    #     PROMPT = ""
    #     # if context != "":
    #     #     PROMPT += "Pay attention and remember information below, which will help to answer the question or imperative after the context ends.\n"
    #     #     PROMPT += context
    #     #     PROMPT += "\nReference the information in the document sources provided within the context above.\n"
    #     PROMPT += "The following is a conversation between a human and an AI. The AI is a helpful assistant. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
    #     PROMPT_TEMPLATE = PromptTemplate(input_variables=['history', 'input'], output_parser=None, partial_variables={}, template=PROMPT, template_format='f-string', validate_template=True)

    #     history_langchain_format = ChatMessageHistory()
    #     for i in range(0, len(history)-1):
    #         (human, ai) = history[i]
    #         history_langchain_format.add_user_message(human)
    #         history_langchain_format.add_ai_message(ai)
    #     openai_memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format)
    #     openai_conversation = ConversationChain(
    #         llm=gpt3_llm,
    #         memory=openai_memory,
    #         prompt=PROMPT_TEMPLATE,
    #     )

    #     bot_message = openai_conversation.run(history[-1][0])
    #     history[-1][1] = bot_message #.split("AI: ")[1]
    #     yield history
    


    #gr.Markdown("OpenProBono")
    with gr.Row():
        # async_chat = gr.Chatbot(
        #     [],
        #     elem_id="llama2-7b-chat",
        #     label="llama2-7b-chat",
        #     #bubble_full_width=True,
        #     #avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        # )
        # sage_chat = gr.Chatbot(
        #     [],
        #     elem_id="sagemaker-llama",
        #     label="llama2-7b",
        #     #bubble_full_width=True,
        #     #avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        # )
        openai_chat = gr.Chatbot(
            [],
            elem_id="OpenProBono",
            label="OpenProBono",
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



    # txt_msg = txt.submit(add_text, [sage_chat, txt], [sage_chat, txt], queue=False).then(
    #     bot, [sage_chat, contxt], sage_chat
    # )
    # txt_msg = txt.submit(add_text, [async_chat, txt], [async_chat, txt], queue=False).then(
    #     async_bot, [async_chat, contxt], async_chat
    # )
    txt_msg = txt.submit(add_text, [openai_chat, txt], [openai_chat, txt], queue=False).then(
        openai_bot, [openai_chat], openai_chat
    )

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    # sub_msg = subbtn.click(add_text, [sage_chat, txt], [sage_chat, txt], queue=False).then(
    #     bot, [sage_chat, contxt], sage_chat
    # )
    # sub_msg = subbtn.click(add_text, [async_chat, txt], [async_chat, txt], queue=False).then(
    #     async_bot, [async_chat, contxt], async_chat
    # )
    sub_msg = subbtn.click(add_text, [openai_chat, txt], [openai_chat, txt], queue=False).then(
        openai_bot, [openai_chat], openai_chat
    )

    sub_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    
demo.queue()

demo.launch(root_path="/",favicon_path="./missing.ico")
