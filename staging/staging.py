import gradio as gr
import random
import time
import os
import json
import logging
import re

from langchain.prompts import BaseChatPromptTemplate
from langchain.vectorstores import Vectara
from langchain.vectorstores.vectara import VectaraRetriever
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish, AIMessage, HumanMessage
from langchain.document_loaders import TextLoader, UnstructuredURLLoader
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain, ConversationChain, LLMChain, LLMCheckerChain

from typing import Dict, List, Union

from langchain import PromptTemplate
from langchain.llms.sagemaker_endpoint import LLMContentHandler

# parts of a model: chat, bot
# - chat is the actualy chat history / output on the screen
# - bot calls the llm endpoint with some prompts and context and langchain magic

from abc import abstractmethod
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
import time, os, uuid

from langchain.agents import AgentExecutor, AgentType, initialize_agent, Tool, ZeroShotAgent, AgentOutputParser, LLMSingleActionAgent
from langchain.llms import OpenAI
from langchain.prompts import MessagesPlaceholder

import langchain

#makes it so it outputs the whole convo and thought process of bot
langchain.debug = True

from serpapi import GoogleSearch
from constants import serp_api_key

GoogleSearch.SERP_API_KEY = serp_api_key

#this function filters the huge dict that serpapi returns to only include the results we want (not ads, etc)
def filtered_search(results):
    print(results)
    print('^^ search results ^^')
    new_dict = {}
    # if('sports_results' in results):
    #     new_dict['sports_results'] = results['sports_results']
    if('organic_results' in results):
        new_dict['organic_results'] = results['organic_results']
        for result in new_dict["organic_results"]:
            result.pop("displayed_link", None)
            result.pop("favicon", None)
            result.pop("about_page_link", None)
            result.pop("about_page_serpapi_link", None)

            summary_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-16k-0613')
            llm_input = """Summarize this web page in less than 100 words.

            Web Page:
            """
            llm_input += str(UnstructuredURLLoader(urls=[result["link"]]).load())
            # summary_chain = LLMChain(llm=summary_llm, prompt=summary_prompt)
            result["full_text"] = summary_llm.predict(llm_input)
    return new_dict

#this is the function that actually calls the serpapi library, with our whitelisted legal sites
def gov_search(q):
    return filtered_search(GoogleSearch({
        'q': "site:*.gov | site:*scholar.google.com | site:*case.law | site:*findlaw.com " + q,
        'num': 5
        }).get_dict())

#this is the function that actually calls the serpapi library, just with a general search
def general_search(q):
    return filtered_search(GoogleSearch({
        'q': q,
        'num': 5
        }).get_dict())

#defining the tools available to the agent
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

tool_names = [tool.name for tool in tools]

# Set up the base template
template = """Complete the user's request as best you can. You have access to the following tools:

{tools}

The following is the chat history so far:
{memory}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, including your sources

These were previous tasks you completed:



Begin!

{input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "memory"]
)

output_parser = CustomOutputParser()

# system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
# too much safety, hurts accuracy

ga_script = """
async () => {
    const script = document.createElement("script");
    script.onload = () =>  console.log("tag manager loaded") ;
    script.src = "https://www.googletagmanager.com/gtag/js?id=G-MKDNM9G2PQ";
    document.head.appendChild(script);

    const script2 = document.createElement("script");
    script2.onload = () => {
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-MKDNM9G2PQ');
    }
    document.head.appendChild(script2);
}
"""

with gr.Blocks(title="OpenProBono",
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.indigo, 
        secondary_hue=gr.themes.colors.blue,
        font=gr.themes.GoogleFont("Open Sans"),
        radius_size=gr.themes.sizes.radius_lg),
    css="footer {visibility: hidden}",
    analytics_enabled=False
    ) as demo:

    #here is where we actually define our llm
    gpt3_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613')

    #need a better way to store emails
    def print_email(email):
        print(email)
        print("^^ this is the email ^^")
        return email
    
    #updates the chat history
    def add_text(history, text):
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)

    #not being used currently, was part of file upload
    def add_file(history, file):
        loader = TextLoader(file.name)
        file_text = loader.load()
        history = history + [((file.name,), None)]
        return history

    #this is the meat, where the chat history is passed and the bot responds
    def openai_bot(history):
        #defining the memory of the agent
        history_langchain_format = ChatMessageHistory()
        for i in range(0, len(history)-1):
            (human, ai) = history[i]
            history_langchain_format.add_user_message(human)
            history_langchain_format.add_ai_message(ai)
        memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")

        #system prompt
        system_message = 'You are a helpful AI assistant. Make a plan to help the user, then execute it. If you cannot answer a question in one step, try to break it down into parts. '
        #system_message += user_prompt
        system_message += '. ALWAYS return a "SOURCES" part in your answer.'

        #initializing the agent
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }
        llm_chain = LLMChain(llm=gpt3_llm, prompt=prompt)
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        # agent = initialize_agent(
        #     tools=tools,
        #     llm=llm_chain,
        #     agent=AgentType.OPENAI_FUNCTIONS,
        #     verbose=True,
        #     agent_kwargs=agent_kwargs,
        #     memory=memory,
        #     max_tokens_limit=4000,
        # )
        #updating the system prompt
        # agent.agent.prompt.messages[0].content = system_message

        #debug
        # print(agent.agent.prompt)
        # print("^^ this agent here^^")
        
        #running the agent and update the history with the response
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)
        bot_message = agent_executor.run(history[-1][0])
        history[-1][1] = bot_message
        yield history
    

    #From here on is the gradio stuff, defining the layout of the page
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

    #corresponds to enter in the text box
    txt_msg = txt.submit(add_text, [openai_chat, txt], [openai_chat, txt], queue=False).then(
        openai_bot, [openai_chat], openai_chat
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    
    #corresponds to clicking the submit button
    sub_msg = subbtn.click(add_text, [openai_chat, txt], [openai_chat, txt], queue=False, api_name="submit").then(
        openai_bot, [openai_chat], openai_chat
    )
    sub_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    #hitting enter and clicking submit for email
    email_txt = emailtxt.submit(print_email, [emailtxt], [emailtxt], queue=False)
    email_msg = emailbtn.click(print_email, [emailtxt], [emailtxt], queue=False)
    demo.load(None, None, None, _js=ga_script)
    
demo.queue()

demo.launch(share=True,favicon_path="./missing.ico")
