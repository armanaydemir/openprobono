import gradio as gr
import random
import time
import os
import json
import sagemaker

from langchain.vectorstores import Vectara, Chroma
from langchain.vectorstores.vectara import VectaraRetriever
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain, ConversationChain, LLMChain, LLMCheckerChain

from typing import Dict

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.prompts import MessagesPlaceholder
from langchain.llms.sagemaker_endpoint import LLMContentHandler

# parts of a model: chat, bot
# - chat is the actualy chat history / output on the screen
# - bot calls the llm endpoint with some prompts and context and langchain magic

from abc import abstractmethod
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint
import boto3, time, os, uuid
from botocore.exceptions import ClientError

from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentExecutor, AgentType, initialize_agent, Tool, ZeroShotAgent
from langchain.llms import OpenAI

#web retriever
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.utilities import GoogleSearchAPIWrapper

import langchain

langchain.debug = True

search = SerpAPIWrapper()
def gov_search(q):
    return search.run("site:*.gov " + q)



system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
# too much safety, hurts accuracy


with gr.Blocks(title="Workspace",
    #font=gr.themes.GoogleFont("Open Sans"),
    css="footer {visibility: hidden}") as demo:

    # Vectorstore
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),persist_directory="./chroma_db_oai")

    # Search 
    search = GoogleSearchAPIWrapper()

    gpt3_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613')

    # Initialize
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=gpt3_llm, 
        search=search
    )
    
    def add_text(history, text):
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)


    def add_file(history, file):
        loader = TextLoader(file.name)
        file_text = loader.load()

        history = history + [((file.name,), None)]
        return history

    def web_research_bot(user_input):
        # history_langchain_format = ChatMessageHistory()
        # for i in range(0, len(history)-1):
        #     (human, ai) = history[i]
        #     history_langchain_format.add_user_message(human)
        #     history_langchain_format.add_ai_message(ai)
        # memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory", input_key='question', output_key='answer')


        import logging
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
        from langchain.chains import RetrievalQAWithSourcesChain
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=gpt3_llm,
            retriever=web_research_retriever,
        )
        result = qa_chain({"question": user_input})
        return result['answer'] + '\n' + result['sources']

    def openai_bot(history, context, user_prompt):
        tools = [
            # Tool(
            #     name="search",
            #     func=search.run,
            #     description="useful for when you need to answer questions about current events. You should ask targeted questions. Always cite your sources.",
            # ),
            # Tool(
            #     name="government-search",
            #     func=gov_search,
            #     description="useful for when you need to answer questions or find resources about government and laws. Always cite your sources.",
            # ),
            Tool(
                name="search",
                func=web_research_retriever.get_relevant_documents,
                description="useful for when you need to answer questions you about recent events. You should ask targeted questions. Always cite your sources.",
            ),
        ]
        history_langchain_format = ChatMessageHistory()
        for i in range(0, len(history)-1):
            (human, ai) = history[i]
            history_langchain_format.add_user_message(human)
            history_langchain_format.add_ai_message(ai)
        memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")
        
        #system_message = 'You are a helpful AI assistant. Always cite your sources. If you do not have enough information to answer a question, ask the user to provide what you need, such as where the user is located.'
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            #"system_message": system_message,
        }
        agent = initialize_agent(
            tools=tools,
            llm=gpt3_llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )
        print(agent)
        bot_message = agent.run(history[-1][0])
        history[-1][1] = bot_message
        yield history
    

    with gr.Row():
        openai_chat = gr.Chatbot(
            [],
            elem_id="OpenProBono",
            label="OpenProBono",
            #bubble_full_width=True,
            #avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        )

    with gr.Row():
        contxt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter any context you want the AI to reference", #, or upload an image",
            container=False,
        )
        user_prompt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter any additional prompt prefix for the AI", #, or upload an image",
            container=False,
        )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter query", #, or upload an image",
            container=False,
        )
        subbtn = gr.Button("Submit")
        #btn = gr.UploadButton("📁", file_types=["text"])

    with gr.Row():
        clearopenai = gr.ClearButton([txt, openai_chat])


    txt_msg = txt.submit(add_text, [openai_chat, txt], [openai_chat, txt], queue=False).then(
        openai_bot, [openai_chat, contxt, user_prompt], openai_chat
    )

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    sub_msg = subbtn.click(add_text, [openai_chat, txt], [openai_chat, txt], queue=False).then(
        openai_bot, [openai_chat, contxt, user_prompt], openai_chat
    )

    sub_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    
    # file_msg = btn.upload(add_file, [openai, btn], [openai], queue=False).then(
    #     bot, openai, openai
    # )
    with gr.Accordion("This is my workspace where I am doing live iterations."):
        gr.Markdown("This demo is a beta meant for informational purposes, demonstrating the abilities of our current technology and to compare different variations of models, prompting methods, document upload, and other features as we continually improve. The data sent in the demo is not guaranteed to be kept private. We will keep iterating on this demo, so keep an eye out for frequent updates.")

demo.queue()

demo.launch(root_path="/wip",server_port=7861,favicon_path="./missing.ico")
