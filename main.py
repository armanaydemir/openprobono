#fastapi implementation of youtube bot
import uuid

import firebase_admin
import langchain
from typing import Annotated, List, Union, Tuple
from fastapi import Body, FastAPI
from firebase_admin import credentials, firestore
import gradio as gr
from langchain import PromptTemplate
from langchain.agents import (AgentExecutor, AgentOutputParser, AgentType,
                              LLMSingleActionAgent, Tool, ZeroShotAgent,
                              initialize_agent)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import \
    FinalStreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (TextLoader, UnstructuredURLLoader,
                                        YoutubeLoader)
from langchain.document_loaders.blob_loaders.youtube_audio import \
    YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import (BaseChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.schema import AgentAction, AgentFinish, AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from pydantic import BaseModel

from anyio.from_thread import start_blocking_portal
from queue import Queue
import re
from serpapi.google_search import GoogleSearch

from bot import opb_bot
from bot import youtube_bot


cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
# opb bot db root path has api prefix
root_path = 'api_'
# manually set api key for now
GoogleSearch.SERP_API_KEY = 'e6e9a37144cdd3e3e40634f60ef69c1ea6e330dfa0d0cde58991aa2552fff980'

def get_uuid_id():
    return str(uuid.uuid4())

def store_conversation(conversation, bot_id, youtube_urls, tools, user_prompt, session, api_key):
    (human, ai) = conversation[-1]
    t = firestore.SERVER_TIMESTAMP
    if(session is None or session == ""):
        session = get_uuid_id()
    data = {"human": human, "ai": ai, 'user_prompt': user_prompt, 'tools': tools, 'youtube_urls': youtube_urls, 'timestamp': t, 'api_key': api_key, "bot_id":bot_id}
    db.collection(root_path + "conversations").document(session).collection('conversations').document("msg" + str(len(conversation))).set(data)
    db.collection(root_path + "conversations").document(session).set({"last_message_timestamp": t}, merge=True)

def create_bot(bot_id, user_prompt, youtube_urls, tools):
    data = {'user_prompt': user_prompt, 'youtube_urls': youtube_urls, 'tools': tools, 'timestamp':  firestore.SERVER_TIMESTAMP}
    db.collection("all_bots").document(bot_id).set(data)

def load_bot(bot_id):
    bot = db.collection("all_bots").document(bot_id).get()
    if(bot.exists):
        return bot.to_dict()
    else:
        return None

#checks api key, determines which to call (youtube or opb, eventually will be all together)
def process(history, user_prompt, youtube_urls, tools, session, bot_id, api_key):
    #if api key is valid (TODO: change this to a real api key check)
    if(api_key == 'xyz' or api_key == 'gradio' or api_key == 'deniz_key'):
        try:
            warn = ""
            #if bot_id is not provided, create a new bot id
            if bot_id is None or bot_id == "":
                bot_id = get_uuid_id()
                create_bot(bot_id, user_prompt, youtube_urls, tools)
            #if bot_id is provided, load the bot
            else:
                bot = load_bot(bot_id)
                #if bot is not found, create a new bot
                if(bot is None):
                    return {"message": "Failure: No bot found with bot id: " + bot_id}
                #else load bot settings
                else:
                    #if user_prompt or youtube_urls are provided, warn user that they are being ignored
                    if(user_prompt is not None and user_prompt != ""):
                        warn +=  " Warning: user_prompt is ignored because bot_id is provided\n"
                    if(youtube_urls is not None and youtube_urls != []):
                        warn +=  " Warning: youtube_urls is ignored because bot_id is provided\n"
                    if(tools is not None and tools != []):
                        warn +=  " Warning: tools is ignored because bot_id is provided\n"
                    user_prompt = bot['user_prompt']
                    youtube_urls = bot['youtube_urls']
                    tools = bot['tools']

            #ONLY use youtube bot if youtube_urls is not empty
            if(youtube_urls is not None and youtube_urls != []):
                output = youtube_bot(history, bot_id, youtube_urls, user_prompt, session)
            else:
                output = opb_bot(history, bot_id, tools, user_prompt, session)
                
            #store conversation (log the api_key)
            store_conversation(chat, bot_id, youtube_urls, tools, user_prompt, session, api_key)

            #return the chat and the bot_id
            return {"message": "Success" + warn, "output": output, "bot_id": bot_id}
        except:
            return {"message": "Failure: Internal Error"}
    else:
        return {"message": "Invalid API Key"}

# FastAPI 

class BotRequest(BaseModel):
    history: list
    user_prompt: str = ""
    tools: list = []
    youtube_urls: list = []
    session: str = None
    bot_id: str = None
    api_key: str = None

api = FastAPI()

@api.get("/", tags=["General"])
def read_root():
    return {"message": "API is alive"}

helper = """
This is an description of all the parameters that can be used. \n\n history: a list of messages in the conversation. (currently chat history is not working, ignores everything but last user message)
\n\n user_prompt: prompt to use for the bot, will use default if empty. \n\n session: session id, used for analytics/logging conversations, not necessary
\n\n youtube_urls: a list of youtube urls used to create a new bot. \n\n bot_id: a bot id used to call previously created bots \n\n api_key: api key necessary for auth
\n\n
Keys to good response:
- Can use this tool to grab videos from playlist https://www.thetubelab.com/get-all-urls-of-youtube-playlist-channel/
- Make sure videos includes only the youtuber talking, because we are grabbing the youtube generated captions, there is no way to differenciate between voices or background game audio which got captioned
- There maybe mispellings / mistakes in the captions which cannot be avoided, espeically with foreign names/words
- Include many / longer videos to get better results
- BotID saves the parameters for the bot, so you can use the same bot multiple times
    - the two parameters saved are user_prompt and youtube_urls
    - if you pass in a bot_id, it will ignore the both of these parameters
"""

@api.post("/youtube", tags=["Youtube API"])
def youtube_bot_request(request: Annotated[
        BotRequest,
        Body(
            openapi_examples={
                "create new youtube bot": {
                    "summary": "create new youtube bot",
                    "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the new bot_id which was created}",
                    "value": {
                        "history": [["hi", ""]],
                        "youtube_urls":["https://youtu.be/6XEOVaL5a1Q", "https://youtu.be/5Qu-TCVCO3Q"],
                        "api_key":"xyz",
                    },
                },
                "create new youtube bot with custom prompt": {
                    "summary": "create new youtube bot with a custom prompt",
                    "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the new bot_id which was created}",
                    "value": {
                        "history": [["hi", ""]],
                        "user_prompt": "Respond like the youtuber in the context below.",
                        "youtube_urls":["https://youtu.be/6XEOVaL5a1Q", "https://youtu.be/5Qu-TCVCO3Q"],
                        "api_key":"xyz",
                    },
                },
                # "call the zealand bot": {
                #     "summary": "call the zealand bot",
                #     "description": "Use a bot_id to call a bot that has already been created for the youtuber zealand. \n\n  Returns: {message: 'Success', output: ai_reply, bot_id: the bot id}",
                #     "value": {
                #         "history": [["hello there", ""]],
                #         "bot_id": "6e39115b-c771-49af-bb12-4cef3d072b45",
                #         "api_key":"xyz",
                #     },
                # },
                # "call the sirlarr bot": {
                #     "summary": "call the sirlarr bot",
                #     "description": "Use a bot_id to call a bot that has already been created for the youtuber sirlarr. \n\n  Returns: {message: 'Success', output: ai_reply, bot_id: the bot id}",
                #     "value": {
                #         "history": [["hello there", ""]],
                #         "bot_id": "6cd7e23f-8be1-4eb4-b18c-55795eb1aca1",
                #         "api_key":"xyz",
                #     },
                # },
                # "call the offhand disney bot": {
                #     "summary": "call the offhand disney bot",
                #     "description": "Use a bot_id to call a bot that has already been created for the youtuber offhand disney. \n\n  Returns: {message: 'Success', output: ai_reply, bot_id: the bot id}",
                #     "value": {
                #         "history": [["hello there", ""]],
                #         "bot_id": "8368890b-a45e-4dd3-a0ba-03250ea0cf30",
                #         "api_key":"xyz",
                #     },
                # },
                "full descriptions of every parameter": {
                    "summary": "Description and Tips",
                    "description": helper,
                    "value": {
                        "history": [["user message 1", "ai replay 1"], ["user message 2", "ai replay 2"], ["user message 3", "ai replay 3"]],
                        "user_prompt": "prompt to use for the bot, will use the default of \"Respond in the same style as the youtuber in the context below.\" if empty",
                        "session": "session id, used for analytics/logging conversations, not necessary",
                        "tools": "tools to be used my the agent, not used in current version",
                        "youtube_urls": ["url of youtube video", "url of youtube video", "url of youtube video"],
                        "bot_id": "id of bot previously created, if bot_id is passed then youtube_urls and user_prompt are ignored",
                        "api_key": "api key necessary for auth",
                    },
                },
            },
        )]):
    request_dict = request.dict()
    history = request_dict['history']
    user_prompt = request_dict['user_prompt']
    tools = request_dict['tools']
    youtube_urls = request_dict['youtube_urls']
    session = request_dict['session']
    bot_id = request_dict['bot_id']
    api_key = request_dict['api_key']
    return process(history, user_prompt, youtube_urls, tools, session, bot_id, api_key)

@api.post("/bot", tags=["General"])
def bot(request: Annotated[
        BotRequest,
        Body(
            openapi_examples={
                "create new bot": {
                    "summary": "create new bot",
                    "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the new bot_id which was created}",
                    "value": {
                        "history": [["hi", ""]],
                        "api_key":"xyz",
                    },
                },

            },
        )]):
    request_dict = request.dict()
    history = request_dict['history']
    user_prompt = request_dict['user_prompt']
    tools = request_dict['tools']
    youtube_urls = request_dict['youtube_urls']
    session = request_dict['session']
    bot_id = request_dict['bot_id']
    api_key = request_dict['api_key']
    return process(history, user_prompt, youtube_urls, tools, session, bot_id, api_key)
