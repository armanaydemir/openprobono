import gradio as gr
import os


import langchain
from langchain import PromptTemplate
from langchain.agents import AgentExecutor, AgentType, initialize_agent, Tool, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage

from serpapi import GoogleSearch


# two main components: chat, bot
# - "___chat" is the actualy chat history / output on the screen
# - "___bot" is the function which calls the llm endpoint with some prompts and context and langchain magic


#makes it so it logs everything made by langchain basically
langchain.debug = True

#manually set the api key for now
GoogleSearch.SERP_API_KEY = "e6e9a37144cdd3e3e40634f60ef69c1ea6e330dfa0d0cde58991aa2552fff980"


##Search Tool for Google

#General Search (no filters)
def general_search(q):
    return filtered_search(GoogleSearch({
        'q': q,
        'num': 5
        }).get_dict())

#Government Search (filtered on whitelist sites of relialbe sources for government))
def gov_search(q):
    return filtered_search(GoogleSearch({
        'q': "site:*.gov " + q,
        'num': 5
        }).get_dict())

#Filter Search results retured by serpapi to only include relavant results
def filtered_search(results):
    new_dict = {}
    if('sports_results' in results):
        new_dict['sports_results'] = results['sports_results']
    if('organic_results' in results):
        new_dict['organic_results'] = results['organic_results']
    return new_dict

#Definition and descriptions of tools aviailable to the bot
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

with gr.Blocks(
    title = "OpenProBono",
    theme = gr.themes.Default(
        primary_hue = gr.themes.colors.indigo, 
        secondary_hue = gr.themes.colors.blue,
        font = gr.themes.GoogleFont("Open Sans"),
        radius_size =g r.themes.sizes.radius_lg,
    ),
    css = "footer {visibility: hidden}"
    ) as app:

    #definition of llm used for bot
    bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613')

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
            llm=bot_llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )
        agent.agent.prompt.messages[0].content = system_message
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
    
app.queue()

app.launch(root_path="/",favicon_path="./missing.ico")
